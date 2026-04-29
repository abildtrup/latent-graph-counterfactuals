import logging
import os

logger = logging.getLogger(__name__)

import hydra
import torch
from lightning_fabric.utilities.seed import seed_everything

from counterfactual_graph_generation.data.make_dataset import CounterfactualGraphDataModule
from counterfactual_graph_generation.models.model import ModelFactory
from counterfactual_graph_generation.models.pegvae import GNNEncoder
from counterfactual_graph_generation.utils import batched_dense_graph_to_batched_data_object
from counterfactual_graph_generation.methods.generate_counterfactuals import generate_counterfactuals
from counterfactual_graph_generation.methods.trainingset_based_methods import evaluate_nn_baselines, get_inputs_and_latents


def decode_and_classify_from_latent_distribution(decoder: torch.nn.Module,
    classifier: torch.nn.Module,
    list_of_latent_predictions: list,
    device,
    sample=False):
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
        device: Whether to run on GPU

    Returns
        List of dense graph representations

    """
    decoder.eval()
    classifier.eval()
    with torch.no_grad():
        predictions = []
        for _, batch in enumerate(list_of_latent_predictions):
            mu, log_var = batch
            mu = mu.to(device)
            z = GNNEncoder.reparameterization(mu, log_var)
            if sample:
                F_new, B_new, A_new, E_new = decoder.sample(z)
            else:
                F_new, B_new, A_new, E_new = decoder.decode_discrete_graph(z)
            if not classifier.requires_dense:
                g = batched_dense_graph_to_batched_data_object(F_new, B_new, A_new, E_new)
                node_embedding, graph_embedding, logits = classifier(g)
            else:
                node_embedding, graph_embedding, logits = classifier([F_new, B_new, A_new, E_new, None])
            probits = logits.softmax(dim=1)
            y_new = torch.argmax(logits, dim=1)
            predictions.append((F_new, B_new, A_new, E_new, y_new, node_embedding, graph_embedding, logits, z, probits))
    return list(zip(*predictions))

def standard_normal_prior_baseline_results(decoder: torch.nn.Module,
    classifier: torch.nn.Module,
    latent_variables: list,
    device,
    sample_size=10,
    sample=True):
    random_CFs_list = []
    for _ in range(sample_size):
        trivial_latent_variables = [(torch.zeros_like(latent_variable[0]), torch.zeros_like(latent_variable[1])) for latent_variable in latent_variables]
        random_CFs = decode_and_classify_from_latent_distribution(decoder, classifier, trivial_latent_variables, device, sample=sample)
        random_CFs_list.append(random_CFs)
    # produce output
    random_sampled_CFs_list = {
        'random_sampled_CFs_list': [{
            'z': torch.cat(random_CFs_list[i][8]).detach().cpu(),
            'F': torch.cat(random_CFs_list[i][0]).detach().cpu(),
            'B': torch.cat(random_CFs_list[i][1]).detach().cpu(),
            'A': torch.cat(random_CFs_list[i][2]).detach().cpu(),
            'E': torch.cat(random_CFs_list[i][3]).detach().cpu(),
            'y': torch.cat(random_CFs_list[i][4]).view(-1,1).detach().cpu(),
            'node_embedding': torch.cat(random_CFs_list[i][5]).detach().cpu(),
            'graph_embedding': torch.cat(random_CFs_list[i][6]).detach().cpu(),
            'logits': torch.cat(random_CFs_list[i][7]).detach().cpu(),
            'probits': torch.cat(random_CFs_list[i][9]).detach().cpu(),
        } for i in range(sample_size)]
    }
    return random_sampled_CFs_list

@hydra.main(config_path="../config", config_name="config.yaml", version_base="1.2")
def main(cfg):
    # Setup data module
    logger.info("Load data module, prepare- and setup data...")
    cfg['dataset']['dense_data_representation'] = True
    seed_everything(seed=cfg['dataset']['seed'], workers=True)
    data_module = CounterfactualGraphDataModule(**cfg['dataset'])
    data_module.prepare_data()
    data_module.setup()

    dataloader_name = cfg['predicter']['dataloader']
    if dataloader_name == 'train':
        dataloader = data_module.train_dataloader()
    elif dataloader_name == 'val':
        dataloader = data_module.val_dataloader()
    elif dataloader_name == 'test':
        dataloader = data_module.test_dataloader()
    else:
        raise ValueError(f'Dataloader "{dataloader_name}" not supported.')

    # Get dataset name:
    dataset_name = cfg['dataset']['dataset_name']

    # Initialize vae
    path = cfg['predicter'][dataset_name]['vae_checkpoint_path']
    logger.info(f'Load model from: {path}')
    vae = ModelFactory.load_model_from_checkpoint(path, model_name=cfg['predicter']['model_name'])
    logger.info("Model was loaded and setup succcesfully.")

    # Initialize classifier
    path = cfg['predicter'][dataset_name]['classifier_checkpoint_path']
    logger.info(f'Load model from: {path}')
    classifier = ModelFactory.load_model_from_checkpoint(path, model_name='GraphClassifier')
    logger.info("Model was loaded and setup succcesfully.")

    # Move model to device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    classifier = classifier.to(device)

    # Get all inputs
    logger.info("Getting inputs and latents...")
    inputs, latent_variables = get_inputs_and_latents(vae.model.encoder, classifier, dataloader, device)
    input_dict = {
            'z': [ None ]*torch.cat(inputs[0]).shape[0], # Not to be used
            'F': torch.cat(inputs[0]).detach().cpu(),
            'B': torch.cat(inputs[1]).detach().cpu(),
            'A': torch.cat(inputs[2]).detach().cpu(),
            'E': torch.cat(inputs[3]).detach().cpu(),
            'y': torch.cat(inputs[4]).detach().cpu(),
            'node_embedding': torch.cat(inputs[5]).detach().cpu(),
            'graph_embedding': torch.cat(inputs[6]).detach().cpu(),
            'logits': torch.cat(inputs[7]).detach().cpu(),
            'probits': torch.cat(inputs[8]).detach().cpu(),
            'mu': torch.cat([item[0] for item in latent_variables]).detach(),
            'log_var': torch.cat([item[1] for item in latent_variables]).detach()
        }

    # Get decoded reconstructions - renconstructions are samples from p(g|z) where and z is sampled from q(z|g)
    logger.info("Decode latents variables...")
    reconstructions = decode_and_classify_from_latent_distribution(vae.model.decoder, classifier, latent_variables, device, sample=True)
    reconstrucion_dict = {
            'z': torch.cat(reconstructions[8]).detach().cpu(),
            'F': torch.cat(reconstructions[0]).detach().cpu(),
            'B': torch.cat(reconstructions[1]).detach().cpu(),
            'A': torch.cat(reconstructions[2]).detach().cpu(),
            'E': torch.cat(reconstructions[3]).detach().cpu(),
            'y': torch.cat(reconstructions[4]).view(-1,1).detach().cpu(),
            'node_embedding': torch.cat(reconstructions[5]).detach().cpu(),
            'graph_embedding': torch.cat(reconstructions[6]).detach().cpu(),
            'logits': torch.cat(reconstructions[7]).detach().cpu(),
            'probits': torch.cat(reconstructions[9]).detach().cpu(),
        }

    # Number of counterfactual examples to generate:
    n = cfg['predicter']['sample_size']

    # Generate n random samples from p(g|z) where z is sampled from p(z):
    logger.info(f'Baseline: Sample {n} random CF graphs from prior...')
    random_sampled_CFs_list = standard_normal_prior_baseline_results(vae.model.decoder, classifier, latent_variables, device, sample_size=n)

    # Generate from nearest neighbour based methods
    logger.info(f'Baseline: Sample {n} random CF graphs using nearest neighbour methods...')
    nn_baseline_results = evaluate_nn_baselines(vae=vae, classifier=classifier, train_dataloader=data_module.train_dataloader(), test_dataloader=dataloader, device=device, n=n, **cfg['predicter']['nn_configuration'])

    # Generate main method CFs:
    logger.info(f'Main method: Sample {n} random CF graphs using classifier guided CFs...')
    cg_point_counterfactuals = generate_counterfactuals(vae=vae.model, classifier=classifier.model, dataloader=dataloader, device=device, sample_size=n, **cfg['predicter']['main_method_configuration'])

    # Create results dict
    counterfactuals = {}
    counterfactuals.update(cg_point_counterfactuals)
    counterfactuals.update(random_sampled_CFs_list)
    counterfactuals.update(nn_baseline_results)

    # Save predicted latent variables in the data/latent_codes folder associated with the dataset.
    dictionary = {
        'input': input_dict,
        'reconstructions': reconstrucion_dict,
        'counterfactuals': counterfactuals
    }

    # Save data
    name = cfg['dataset']['dataset_name']
    path = os.path.join('data/predictions' , f'{dataloader_name}_{name}.pt')
    logger.info(f'Saving data at: {path}')
    torch.save(dictionary, path)
    return

if __name__ == '__main__':
    main()
