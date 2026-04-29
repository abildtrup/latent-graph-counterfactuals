import logging

logger = logging.getLogger(__name__)

import hydra
import wandb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
from torch import Tensor
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch_geometric.data import Data

from counterfactual_graph_generation.models.graphClassifier import GraphClassifier, PossibleGraphClassifiersEnum
from counterfactual_graph_generation.models.pegvae import PEGVAE, AdaptedPEGVAE
from counterfactual_graph_generation.models.adapter import Adapter
from counterfactual_graph_generation.visualizations.plotting import plot_roc_curve, plot_getometric_mean

def data_batch_to_dense_batch(batch):
    if isinstance(batch, Data):
            F, B, A, E, _ = batch.x, batch.b, batch.adj, batch.e, batch.y
    else:
            F, B, A, E, _ = batch
    return F, B, A, E, _

class ClassifierGuidedCF(torch.nn.Module):
    def __init__(self, z, y, decoder, classifier, lr, lamb: float, tau: float):
        super().__init__()

        self.tau = tau
        self.lamb = lamb
        self.lr = lr
        self.decoder = decoder
        self.classifier = classifier
        self.z_w = nn.Parameter(z.data.clone()) # N x 1 x V
        self.z = z.data.clone()
        self.input_label = y.view(-1)
        self.desired_label = 1 - self.input_label

        self.optimizer = ClassifierGuidedCF.configure_optimizers(self.z_w, self.lr)['optimizer']

    def forward(self, ):
        F, B, A, E = self.decoder.decode_discrete_graph(self.z_w, self.tau)
        node_embeddings, graph_embedding, logits = self.classifier([F, B, A, E, None])
        return self.z_w, F, B, A, E, node_embeddings, graph_embedding, logits

    def training_step(self,):
        out = self()
        loss = torch.nn.functional.nll_loss(out[-1], self.desired_label, reduction='none')
        loss += (self.lamb * torch.norm(out[0], dim=2)).view(-1)
        # Set loss to zero is label has flipped
        y_sample_pred = torch.argmax(out[-1], dim=1).view(-1)
        y_mask = 1-(self.desired_label == y_sample_pred).float()
        loss = loss * y_mask
        return loss.mean(), out

    def train_model(self, n=20):
        train_info = []
        for _ in range(n):
            self.optimizer.zero_grad()
            loss, out = self.training_step()
            # Log train info:
            train_info.append((self.z_w.data.clone(), loss.item()))
            # Backward
            loss.backward()
            self.optimizer.step()
        zs, loss = list(zip(*train_info))
        return {'zs': torch.stack(zs), 'loss':loss}

    def sample_counterfactual(self,):
        F, B, A, E = self.decoder.sample(self.z_w)
        node_embeddings, graph_embedding, logits = self.classifier([F, B, A, E, None])
        return self.z_w, F, B, A, E, node_embeddings, graph_embedding, logits

    def reset_parameters(self):
        self.z_w = nn.Parameter(self.z.clone())
        self.optimizer = ClassifierGuidedCF.configure_optimizers(self.z_w, self.lr)

    @staticmethod
    def configure_optimizers(z_w, lr):
        learnable_parameters = list([z_w])
        optimizer = torch.optim.Adam(learnable_parameters, lr=lr)
        return {'optimizer':optimizer}


class PLGraphVAE(pl.LightningModule):
    def __init__(self,
            model_name: str,
            pl_config: dict,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.batch_size = pl_config['batch_size']
        self.pl_config = pl_config
        self.model = PEGVAE(**pl_config['model_config'])

        # Loss functions
        self.bce_loss_with_logits = BCEWithLogitsLoss(reduction='sum')
        self.bce_loss = BCELoss(reduction='sum')
        self.ce_loss = CrossEntropyLoss(reduction='sum')

        # Updates before KL-loss
        self.configure_kl_loss(**pl_config['kl_config'])

        # Metrics
        self.test_ROC_binary_B = torchmetrics.classification.BinaryROC()
        self.test_AUROC_binary_B = torchmetrics.classification.BinaryAUROC()
        self.test_ROC_binary_A = torchmetrics.classification.BinaryROC()
        self.test_AUROC_binary_A = torchmetrics.classification.BinaryAUROC()
        #self.test_ROC_nodes = torchmetrics.classification.MulticlassROC(num_classes=pl_config['model_config']['encoder_config']['node_features'])
        #self.test_AUROC_nodes = torchmetrics.classification.MulticlassAUROC(num_classes=pl_config['model_config']['encoder_config']['node_features'])

    def forward(self, batch):
        F, B, A, E, _ = batch
        return self.model.forward(F, B, A, E)

    def training_step(self, batch, batch_idx):
        batch = data_batch_to_dense_batch(batch) # TODO: Make redundant
        out = self(batch)
        reconstruction_loss, _, _, _, _= self.reconstruction_loss(batch, out)
        kl_loss = self.kl_loss(out)
        loss = reconstruction_loss + min(self.kl_weight, self.max_kl_weight) * kl_loss
        if self.start_kl_loss_phasing <= 0:
            self.kl_weight += self.kl_weight_step
        self.start_kl_loss_phasing -= 1
        self.log("reconstruction_train", float(reconstruction_loss), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("kl_train", float(kl_loss), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("ELBO_train", float(reconstruction_loss) + float(kl_loss), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("train_loss", float(loss), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = data_batch_to_dense_batch(batch) # TODO: Make redundant
        out = self(batch)
        reconstruction_loss, b_loss, f_loss, a_loss, e_loss= self.reconstruction_loss(batch, out)
        kl_loss = self.kl_loss(out)
        adjusted_loss = reconstruction_loss + (min(self.kl_weight, self.max_kl_weight) * kl_loss)
        monitored_loss = reconstruction_loss + self.max_kl_weight * kl_loss
        self.log("reconstruction_val", float(reconstruction_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("B_reconstruction_val", float(b_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("F_reconstruction_val", float(f_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("A_reconstruction_val", float(a_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("E_reconstruction_val", float(e_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("kl_val", float(kl_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("ELBO_val", float(reconstruction_loss) + float(kl_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val_loss_adjusted", float(adjusted_loss) , on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        if self.kl_weight_step == 0:
            self.log("val_loss", float(adjusted_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        else:
            self.log("val_loss", float(monitored_loss), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        if self.lr_schedulers().last_epoch != 0:
            self.log("Learning rate", self.lr_schedulers().get_last_lr()[-1], on_step=False, on_epoch=True, batch_size=self.batch_size)
        return monitored_loss

    def get_plotting_statistics_binary(self, roc_metric, auroc_metric):
        fpr, tpr, thresholds = roc_metric.compute()
        roc_auc = auroc_metric.compute()
        # To cpu
        fpr, tpr, thresholds = fpr.cpu(), tpr.cpu(), thresholds.cpu()
        roc_auc = roc_auc.cpu()
        # Compute paramters
        geometric_mean = torch.sqrt((1-fpr) * tpr)
        idx = geometric_mean.argmax().item()
        roc_curve_parameters = [fpr, tpr, thresholds, roc_auc, idx]
        geometric_mean_parameters = [geometric_mean, thresholds, idx]
        return roc_curve_parameters, geometric_mean_parameters, thresholds[idx]

    def test_step(self, batch, batch_idx, dataloader_idx):
        batch = data_batch_to_dense_batch(batch) # TODO: Make redundant
        out = self(batch)
        reconstruction_loss, b_loss, f_loss, a_loss, e_loss= self.reconstruction_loss(batch, out)
        kl_loss = self.kl_loss(out)
        loss = reconstruction_loss + (min(self.kl_weight, self.max_kl_weight) * kl_loss)
        ELBO = reconstruction_loss + kl_loss
        # Losses for main paper:
        self.log("ELBO", float(ELBO), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("Reconstruction loss", float(reconstruction_loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("KL loss", float(kl_loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("loss", float(loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        # Additional losses of interest:
        self.log("Reconstruction loss (B)", float(b_loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("Reconstruction loss (F)", float(f_loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("Reconstruction loss (A)", float(a_loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        self.log("Reconstruction loss (E)", float(e_loss), on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        if dataloader_idx == 1: # Plot validation curves for model selection
            # Estract B
            B_pred, B_target = out[0][1], batch[1]
            self.test_ROC_binary_B.update(preds=B_pred, target=B_target.long())
            self.test_AUROC_binary_B.update(preds=B_pred, target=B_target.long())
            # Extract A
            A_pred, A_target = out[0][2], batch[2]
            A_mask = (B_target.permute(0,2,1) @ B_target).bool() #A_mask = (B_pred_thresholded.permute(0,2,1) @ B_pred_thresholded).bool() + (B_target.permute(0,2,1) @ B_target).bool() # Note: Mask includes all true negative edges aswell as false postive edges
            masked_pred_A = A_pred.view(-1)[A_mask.view(-1)]
            masked_target_A = A_target.view(-1)[A_mask.view(-1)]
            self.test_ROC_binary_A.update(preds=masked_pred_A, target=masked_target_A.long())
            self.test_AUROC_binary_A.update(preds=masked_pred_A, target=masked_target_A.long())
        return loss

    def on_test_end(self,):
        statistics_B = self.get_plotting_statistics_binary(self.test_ROC_binary_B, self.test_AUROC_binary_B)
        statistics_A = self.get_plotting_statistics_binary(self.test_ROC_binary_A, self.test_AUROC_binary_A)
        # Plot ROC
        wandb.log({"ROC (B)": plot_roc_curve(*statistics_B[0])})
        wandb.log({"Geometric Mean (B)": plot_getometric_mean(*statistics_B[1])})
        wandb.log({"ROC (A)": plot_roc_curve(*statistics_A[0])})
        wandb.log({"Geometric Mean (A)": plot_getometric_mean(*statistics_A[1])})

    def configure_optimizers(self):
        learnable_parameters = list(self.model.parameters())
        optimizer = torch.optim.Adam(learnable_parameters, lr=self.pl_config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=150, cooldown=25)
        return {'optimizer':optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}}

    def reconstruction_loss(self, batch_in, batch_out):
        F, B, A, E, _ = batch_in
        (F_new, B_new, A_new, E_new), _ = batch_out

        batch_size = F.shape[0]
        # Loss on the binary indicator:
        b_loss = self.bce_loss_with_logits(input=B_new, target=B) / batch_size
        # Loss on the node features given the true binary indicator
        f_loss = self.ce_loss(input=F_new * B, target=F) / batch_size
        # Loss on the binary adjacency matrix given known graphsize and node features:
        a_loss = self.bce_loss(input=A_new.sigmoid() * (B.permute(0,2,1) @ B).unsqueeze(dim=1), target=A) / batch_size
        # Loss on the edge features given everything else is known. Is binary if number of edge features is 0
        e_loss = self.ce_loss(input=E_new * A, target=E) / batch_size
        loss = b_loss + f_loss + a_loss + e_loss
        return loss.mean(dim=0), b_loss.mean(dim=0), f_loss.mean(dim=0), a_loss.mean(dim=0), e_loss.mean(dim=0)

    # KL loss:
    def kl_loss(self, batch_out):
        _, (mu_e, log_var_e) = batch_out
        z_lsgms, z_mu = 0.5*log_var_e, mu_e
        kl_loss = -torch.mean(0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp(), dim = (1,2)))
        kl_loss = torch.clamp(kl_loss, max=1e12)
        return kl_loss

    def configure_kl_loss(self, start_kl_loss_phasing, kl_weight, max_kl_weight, kl_weight_step):
        self.start_kl_loss_phasing = start_kl_loss_phasing #12000
        self.kl_weight = kl_weight #0
        self.max_kl_weight = max_kl_weight #0.05
        self.kl_weight_step = kl_weight_step # 1.25e-06


class PLGraphVAEwithAdapter(PLGraphVAE):
    def __init__(self,
            model_name: str,
            pl_config: dict,
        ):
        pretrained_model = ModelFactory.load_model_from_checkpoint(**pl_config.pretrained_model_config)
        pretrained_model.model.decoder.delete_last_layer()
        super().__init__(model_name, pretrained_model.pl_config)
        self.save_hyperparameters(ignore=['pretrained_model'])

        self.model_name = model_name
        self.encoder_adapter = Adapter(pl_config.model_config.encoder_adapter) # Should point to the adapter
        self.decoder_adapter = Adapter(pl_config.model_config.decoder_adapter)
        self.model = AdaptedPEGVAE(pretrained_model.model, self.encoder_adapter, self.decoder_adapter)

        # Update pl-config to new pytorch lighntnig config
        self.pl_config = pl_config
        self.batch_size = pl_config['batch_size']
        self.configure_kl_loss(**pl_config['kl_config'])

    def forward(self, batch):
        F, B, A, E, _ = batch
        return self.model.forward(F, B, A, E)

    def configure_optimizers(self):
        learnable_parameters = list(self.encoder_adapter.parameters()) + list(self.decoder_adapter.parameters())
        optimizer = torch.optim.Adam(learnable_parameters, lr=self.pl_config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=150, cooldown=25)
        return {'optimizer':optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}}


class PLGraphClassifier(pl.LightningModule):
    def __init__(self,
            model_name: str,
            pl_config: dict,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.batch_size = pl_config['batch_size']
        self.pl_config = pl_config
        self.requires_dense = pl_config['requires_dense']
        self.type = pl_config['type']
        try:
            self.model = PossibleGraphClassifiersEnum[self.type].value(**pl_config['model_config'])
        except:
            raise ValueError(f'Model type "{self.type}" not supported. Possible values are "{[e.value for e in PossibleGraphClassifiersEnum]}"')

        # Metrics
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=pl_config['model_config']['num_classes'], average='weighted')
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=pl_config['model_config']['num_classes'], average='weighted')
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=pl_config['model_config']['num_classes'])
        self.train_AUROC = torchmetrics.classification.AUROC(task="multiclass", num_classes=pl_config['model_config']['num_classes'])
        self.valid_AUROC = torchmetrics.classification.AUROC(task="multiclass", num_classes=pl_config['model_config']['num_classes'])
        self.test_AUROC = torchmetrics.classification.AUROC(task="multiclass", num_classes=pl_config['model_config']['num_classes'])

    def forward(self, inpt_batch):
        return self.model.forward(inpt_batch)

    def training_step(self, batch, batch_idx):
        logits = self(batch)[-1]
        if self.requires_dense:
            target = batch[-1].view(-1)
        else:
            target = batch.y
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits), target)
        self.train_AUROC(logits, target)
        self.log("train_loss", float(loss), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("train_auroc", self.train_AUROC, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)[-1]
        if self.requires_dense:
            target = batch[-1].view(-1)
        else:
            target = batch.y
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits), target)
        self.valid_acc(logits, target)
        self.valid_AUROC(logits, target)
        self.log("val_loss", float(loss), on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val_accuracy", self.valid_acc(logits, target), on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val_auroc", self.valid_AUROC(logits, target), on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def on_validation_end(self) -> None:
        return super().on_validation_end()

    def test_step(self, batch, batch_idx, dataloader_idx):
        logits = self(batch)[-1]
        if self.requires_dense:
            target = batch[-1].view(-1)
        else:
            target = batch.y
        loss = torch.nn.functional.nll_loss(logits, target)
        accuracy = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits), target)
        self.test_f1(logits, target)
        self.test_AUROC(logits, target)
        self.log("loss", float(loss), on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        self.log("accuracy", accuracy, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        self.log("f1", self.test_f1, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        self.log("test_auroc", self.test_AUROC, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        learnable_parameters = list(self.model.parameters())
        optimizer = torch.optim.Adam(learnable_parameters, lr=self.pl_config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, cooldown=10)
        return {'optimizer':optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}}


class ModelFactory():
    """
    A factory class for creating PyTorch Lightning models.

    Parameters:
    - model_name (str): The name of the model.
    - pl_config (Any): Configuration for the PyTorch Lightning model.

    Attributes:
    - model_name (str): The name of the model.
    - pl_config (Any): Configuration for the PyTorch Lightning model.
    """

    def __init__(self,
            model_name: str,
            pl_config: dict,
            resume_training_from_ckpt: dict = None
        ):
        self.model_name = model_name
        self.pl_config = pl_config
        self.resume_training_from_ckpt: dict = resume_training_from_ckpt

    @staticmethod
    def load_model_from_checkpoint(path, model_name) -> pl.LightningModule:
        if model_name == "GraphClassifier":
            return PLGraphClassifier.load_from_checkpoint(path)
        elif model_name == "PEGVAE":
            return PLGraphVAE.load_from_checkpoint(path)
        elif model_name == "PretrainedPEGVAE":
            return PLGraphVAEwithAdapter.load_from_checkpoint(path)
        else:
            raise ValueError(f'Model name "{model_name}" not supported.')

    def setup_model(self) -> pl.LightningModule:
        if self.model_name == "GraphClassifier":
            return PLGraphClassifier(self.model_name, self.pl_config)
        elif self.model_name == "PEGVAE":
            return PLGraphVAE(self.model_name, self.pl_config)
        elif self.model_name == "PretrainedPEGVAE":
            return PLGraphVAEwithAdapter(self.model_name, self.pl_config)
        elif self.resume_training_from_ckpt  != None:
            return self.load_model_from_checkpoint(**self.resume_training_from_ckpt)
        else:
            raise ValueError(f'Model name "{self.model_name}" not supported.')




@hydra.main(config_path="../../config/model", config_name="AidsPretrainedPEGVAE.yaml", version_base="1.2")
def main(cfg):
    logger.info("Initialize PyTorch Lightning model from default configurations...")
    model_factory = ModelFactory(**cfg)
    logger.info("Setup model...")
    model = model_factory.setup_model()
    logger.info("Model was loaded and setup succcesfully.")
    return model

if __name__ == '__main__':
    main()
