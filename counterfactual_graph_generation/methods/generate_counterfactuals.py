import logging
import wandb

logger = logging.getLogger(__name__)

import hydra
import omegaconf
import torch
from lightning_fabric.utilities.seed import seed_everything

from counterfactual_graph_generation.data.make_dataset import CounterfactualGraphDataModule
from counterfactual_graph_generation.models.model import ModelFactory, ClassifierGuidedCF
from counterfactual_graph_generation.eval import flip_ratio, latent_distance
from counterfactual_graph_generation.methods.trainingset_based_methods import get_inputs_and_latents

def generate_counterfactuals(vae, classifier, dataloader, device, sample_size=10, max_iterations=20, lr=0.001, lamb=0.0, sample_counterfactual=False, tau=1):
    vae.decoder.eval()
    vae.encoder.eval()
    classifier.eval()
    cfs = [[] for _ in range(sample_size)]
    for _, batch in enumerate(dataloader):
        F, B, A, E, y = batch
        F, B, A, E, y = F.to(device), B.to(device), A.to(device), E.to(device), y.to(device)
        with torch.no_grad():
            mu, _ = vae.encoder.encode(F, B, A, E)
        model = ClassifierGuidedCF(z=mu, y=y, decoder=vae.decoder, classifier=classifier, lr=lr, lamb=lamb, tau=tau)
        train_info = model.train_model(n=max_iterations)
        for i in range(sample_size):
            if sample_counterfactual:
                _, F_sample, B_sample, A_sample, E_sample, node_embeddings, graph_embedding, logits = model.sample_counterfactual()
            else:
                _, F_sample, B_sample, A_sample, E_sample, node_embeddings, graph_embedding, logits = model.forward()
            probits = logits.softmax(dim=1)
            y_sample_pred = torch.argmax(logits, dim=1)
            sample = {
                'z': train_info['zs'][-1].detach().cpu(),
                'F': F_sample.detach().cpu(),
                'B': B_sample.detach().cpu(),
                'A': A_sample.detach().cpu(),
                'E': E_sample.detach().cpu(),
                'y': y_sample_pred.detach().cpu(),
                'node_embedding': node_embeddings.detach().cpu(),
                'graph_embedding': graph_embedding.detach().cpu(),
                'logits': logits.detach().cpu(),
                'probits': probits.detach().cpu(),
                }
            cfs[i].append(sample)
    # Make counterfactuals on correct format
    cfs_reformated = {
        'main_method': [
            {
                'z': torch.cat([cfs[i][j]['z'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'F': torch.cat([cfs[i][j]['F'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'B': torch.cat([cfs[i][j]['B'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'A': torch.cat([cfs[i][j]['A'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'E': torch.cat([cfs[i][j]['E'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'y': torch.cat([cfs[i][j]['y'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'node_embedding': torch.cat([cfs[i][j]['node_embedding'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'graph_embedding': torch.cat([cfs[i][j]['graph_embedding'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'logits': torch.cat([cfs[i][j]['logits'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                'probits': torch.cat([cfs[i][j]['probits'] for j, _ in enumerate(cfs[i])]).detach().to(torch.float16).cpu(),
                #'train_info': [cfs[i][j]['train_info'] for j, _ in enumerate(cfs[i])],
            } for i in range(sample_size)
        ]
    }
    return cfs_reformated

@hydra.main(config_path="../../config", config_name="config.yaml", version_base="1.2")
def main(cfg):
    # Initialize wandb run
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    _ = wandb.init()

    # Setup data module
    logger.info("Load data module, prepare- and setup data...")
    cfg['dataset']['dense_data_representation'] = True
    seed_everything(seed=cfg['dataset']['seed'], workers=True)
    data_module = CounterfactualGraphDataModule(**cfg['dataset'])
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.test_dataloader()
    # Get dataset name:
    dataset_name = cfg['dataset']['dataset_name']
    # Initialize model
    path = cfg['predicter'][dataset_name]['vae_checkpoint_path']
    logger.info(f'Load VAE from: {path}')
    vae = ModelFactory.load_model_from_checkpoint(path, model_name='PEGVAE').model
    logger.info("VAE was loaded and setup succcesfully.")
    # Initialize classifier
    path = cfg['predicter'][dataset_name]['classifier_checkpoint_path']
    logger.info(f'Load Classifier from: {path}')
    classifier = ModelFactory.load_model_from_checkpoint(path, model_name='GraphClassifier')
    logger.info("Classifier was loaded and setup succcesfully.")
    # Move model to device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    vae.eval()
    classifier = classifier.to(device)
    classifier.eval()
    # Intialize
    logger.info("Initialize classifier guided counterfactuals model...")
    counterfactuals = generate_counterfactuals(vae, classifier.model, dataloader, device, sample_size=10, **cfg['predicter']['main_method_configuration'])
    logger.info("Successfully generated counterfactuals")

    # EVALUATION and LOGGING:
    inputs, latents = get_inputs_and_latents(encoder=vae.encoder, classifier=classifier, dataloader=dataloader, device=device)
    # Compute flip ratio
    y_target = 1 - torch.cat(inputs[4]).detach().view(-1)
    y_cf = counterfactuals['main_method'][0]['y'].view(-1)
    fr = flip_ratio(y_target, y_cf)
    # Compute latent distance
    input_z = latents[0][0]
    cf_z = counterfactuals['main_method'][0]['z']
    mean_distance = torch.linalg.vector_norm(input_z - cf_z, dim=2).mean()
    # Log metrics
    wandb.log({
        'flip_ratio': float(fr),
        'latent_distance': float(mean_distance),
        'loss': (1 - float(fr)) / (1 - 0.9)  + float(mean_distance) / 2.5
    })
    return

if __name__ == '__main__':
    main()
