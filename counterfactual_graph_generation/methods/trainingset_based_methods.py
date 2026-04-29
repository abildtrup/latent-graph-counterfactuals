import logging

from counterfactual_graph_generation.utils import batched_dense_graph_to_batched_data_object

logger = logging.getLogger(__name__)

import hydra
import torch
from lightning_fabric.utilities.seed import seed_everything
from counterfactual_graph_generation.data.make_dataset import CounterfactualGraphDataModule
from counterfactual_graph_generation.models.model import ModelFactory
from counterfactual_graph_generation.models.pegvae import GNNEncoder
from counterfactual_graph_generation.invariant_maps.maps import invariant_channel_sort


def get_nearest_neighbour(z_test, z_train, k=40):
    z_train = z_train.squeeze(1)
    zs = []
    for _, z in enumerate(z_test):
        dist = torch.norm(z_train-z, dim=1)
        knn = dist.topk(k, largest=False)
        nearest_z = z_train[knn.indices]
        zs.append(nearest_z.mean(dim=0, keepdim=True))
    return torch.stack(zs, dim=0)

def get_closest_training_point(test, train, k):
    z_test, y_desired = test
    z_train, dense_training_graph = train
    y_train = dense_training_graph[4].view(-1)

    # Split z_train based on y_train- mask
    z_train_positive = z_train[y_train.bool()]
    z_train_negative = z_train[~(y_train.bool())]

    # Calculate nearest neighbours of z_train for each class
    nearest_positive = get_nearest_neighbour(z_test, z_train_positive, k=k)
    nearest_negative = get_nearest_neighbour(z_test, z_train_negative, k=k)

    # Pick the ones from the desired lass
    nearest_neighbour_of_desired_class = nearest_positive * y_desired.view(-1,1,1) + nearest_negative * ((y_desired.view(-1,1,1) + 1)%2)
    return nearest_neighbour_of_desired_class

def get_nearest_neighbour_indices(z_test, z_train):
    z_train = z_train.squeeze(1)
    indices = []
    zs = []
    for _, z in enumerate(z_test):
        dist = torch.norm(z_train-z, dim=1)
        knn = dist.topk(1, largest=False)
        indices.append(knn.indices)
        nearest_z = z_train[knn.indices]
        zs.append(nearest_z.mean(dim=0, keepdim=True))
    return torch.cat(indices).view(1,-1), torch.stack(zs, dim=0)

def get_closest_training_graph(test, train):
    z_test, y_desired = test
    z_train, dense_training_graph = train
    y_train = dense_training_graph[4].view(-1)

    # Split z_train based on y_train- mask
    z_train_positive = z_train[y_train.bool()]
    dense_training_graph_positive = [g[y_train.bool()] for g in dense_training_graph]
    z_train_negative = z_train[~(y_train.bool())]
    dense_training_graph_negative = [g[~(y_train.bool())] for g in dense_training_graph]

    # Calculate nearest neighbours of z_train for each class
    nearest_positive_idx, zs_positive = get_nearest_neighbour_indices(z_test, z_train_positive)
    nearest_negative_idx, zs_negative = get_nearest_neighbour_indices(z_test, z_train_negative)
    nearest_positive_dense_graph = [g[nearest_positive_idx.view(-1)] for g in dense_training_graph_positive]
    nearest_negative_dense_graph = [g[nearest_negative_idx.view(-1)] for g in dense_training_graph_negative]

    # Pick the ones from the desired lass
    nearest_dense_graph = [nearest_positive_dense_graph[i] * y_desired.view(-1,*([1]*(len(nearest_positive_dense_graph[i].shape)-1))) + nearest_negative_dense_graph[i] * ((y_desired.view(-1,*([1]*(len(nearest_negative_dense_graph[i].shape)-1))) + 1)%2) for i, _ in enumerate(nearest_positive_dense_graph)]
    nearest_dense_graph.append(zs_positive * y_desired.view(-1, 1, 1) + zs_negative * ((y_desired.view(-1,1,1) + 1)%2))
    return nearest_dense_graph


def get_closest_invariant_point(z_test, z_train):
    # Ensure that points are on cpu before converting to numpy
    z_test, z_train = z_test.cpu(), z_train.cpu()
    # Sort
    z_test, _, _ = invariant_channel_sort(z_test)
    z_train, _, _ = invariant_channel_sort(z_train)
    # To torch
    z_test = torch.Tensor(z_test)
    z_train = torch.Tensor(z_train)
    # Do similar to get closest
    z_train = z_train.squeeze(1)
    zs = []
    for i, z in enumerate(z_test):
        dist = torch.norm(z_train-z, dim=1)
        knn = dist.topk(1, largest=False)
        nearest_z = z_train[knn.indices]
        zs.append(nearest_z)
    return torch.stack(zs, dim=0)

def sample_z_from_prior():
    return

def sample_z_from_posterior():
    return

def get_inputs_and_latents(encoder: torch.nn.Module,
                           classifier: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device):
    classifier.eval()
    encoder.eval()
    inputs = []
    latents = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            F, B, A, E, y = batch
            F, B, A, E, y = F.to(device), B.to(device), A.to(device), E.to(device), y.to(device)
            if not classifier.requires_dense:
                g = batched_dense_graph_to_batched_data_object(F, B, A, E)
                node_embedding, graph_embedding, logits = classifier(g)
            else:
                node_embedding, graph_embedding, logits = classifier([F, B, A, E, y])
            probits = logits.softmax(dim=1)
            inputs.append([F, B, A, E, y, node_embedding, graph_embedding, logits, probits])
            latents.append(encoder(F, B, A, E))
        Fs, Bs, As, Es, ys, node_embeddings, graph_embeddings, logits, probits = list(zip(*inputs))
    return (Fs, Bs, As, Es, ys, node_embeddings, graph_embeddings, logits, probits), latents

def decode_and_classify_latents(decoder: torch.nn.Module, classifier: torch.nn.Module, list_of_latent_codes: list, device, sample=False):
    decoder.eval()
    classifier.eval()
    with torch.no_grad():
        predictions = []
        for _, z in enumerate(list_of_latent_codes):
            z = z.to(device)
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

def classify_list_of_input_graphs(closest_graphs,
                                  classifier: torch.nn.Module,
                                  device,):
    classifier.eval()
    with torch.no_grad():
        predictions = []
        z = closest_graphs[-1]
        F_new, B_new, A_new, E_new = closest_graphs[0], closest_graphs[1], closest_graphs[2], closest_graphs[3]
        node_embedding, graph_embedding, logits = classifier([F_new, B_new, A_new, E_new, None])
        probits = logits.softmax(dim=1)
        y_new = torch.argmax(logits, dim=1)
        predictions.append((F_new, B_new, A_new, E_new, y_new, node_embedding, graph_embedding, logits, z, probits))
    return list(zip(*predictions))

def produce_output_dictionary(list_of_decoded_graphs):
    output_dict = [{
        'z': torch.cat(list_of_decoded_graphs[i][8]).detach().cpu(),
        'F': torch.cat(list_of_decoded_graphs[i][0]).detach().cpu(),
        'B': torch.cat(list_of_decoded_graphs[i][1]).detach().cpu(),
        'A': torch.cat(list_of_decoded_graphs[i][2]).detach().cpu(),
        'E': torch.cat(list_of_decoded_graphs[i][3]).detach().cpu(),
        'y': torch.cat(list_of_decoded_graphs[i][4]).view(-1,1).detach().cpu(),
        'node_embedding': torch.cat(list_of_decoded_graphs[i][5]).detach().cpu(),
        'graph_embedding': torch.cat(list_of_decoded_graphs[i][6]).detach().cpu(),
        'logits': torch.cat(list_of_decoded_graphs[i][7]).detach().cpu(),
        'probits': torch.cat(list_of_decoded_graphs[i][9]).detach().cpu(),
     } for i, _ in enumerate(list_of_decoded_graphs)]
    return output_dict

def evaluate_nn_baselines(vae: torch.nn.Module,
                          classifier: torch.nn.Module,
                          train_dataloader: torch.utils.data.DataLoader,
                          test_dataloader: torch.utils.data.DataLoader,
                          device,
                          n: int,
                          k: int):
    # Get latent TRAINING datapoints.
    logger.info("Finding latent train-variables...")
    inputs_train, latent_variables_train = get_inputs_and_latents(vae.model.encoder, classifier, train_dataloader, device)
    latent_variables_train = torch.cat([item[0] for item in latent_variables_train]).detach()
    y_dense_graph = [torch.cat(inputs_train[i]) for i in range(5)]
    # Get latent TEST datapoints
    logger.info("Finding latent test-variables...")
    inputs_test, latent_variables_test = get_inputs_and_latents(vae.model.encoder, classifier, test_dataloader, device)
    latent_variables_test = torch.cat([item[0] for item in latent_variables_test]).detach()
    y_test = torch.cat(inputs_test[4]).view(-1)
    y_desired = (y_test + 1) % 2
    # Find closest TRAINING datapoint for each TEST datapoint
    logger.info("Finding closest train point test-variables...")
    closest_z = get_closest_training_point((latent_variables_test, y_desired), (latent_variables_train, y_dense_graph), k=k)
    # Find closest DECODED TRAINING datapoint for each TEST datapoint
    logger.info("Decode and classify latent codes..")
    decoded_closest_z_list = [decode_and_classify_latents(vae.model.decoder, classifier, [closest_z], device, sample=True) for i in range(n)]
    decoded_closest_z_list = produce_output_dictionary(decoded_closest_z_list)
    # Find closest TRUE DENSE GRAPH for each TEST datapoint
    logger.info("Get the closest training true label training graph..")
    closest_graphs = get_closest_training_graph((latent_variables_test, y_desired), (latent_variables_train, y_dense_graph))
    # Find closest INVARIANT TRAINING datapoint
    #logger.info("Finding closest INVARIANT train point test-variables...")
    #closest_invariant_z = get_closest_invariant_point(latent_variables_test, latent_variables_train)
    # Find closest INVARIANT TRAINING datapoint for each TEST datapoint
    #logger.info("Decode and classify INVARIANT latent codes..")
    #decoded_closest_invariant_z_list = [decode_and_classify_latents(vae.model.decoder, classifier, [closest_invariant_z], device, sample=True) for i in range(n)]
    # CLassify closest graph:
    logger.info("Classify training true label training graph..")
    classified_closest_dense_graph = [classify_list_of_input_graphs(closest_graphs=closest_graphs, classifier=classifier, device=device)]
    # Produce method output
    output = {
        'closest_latent_train': decoded_closest_z_list,
        #'closest_invariant_latent_train': produce_output_dictionary(decoded_closest_invariant_z_list),
        'closest_graph_train': produce_output_dictionary(classified_closest_dense_graph),
    }
    return output

@hydra.main(config_path="../../config", config_name="config.yaml", version_base="1.2")
def main(cfg):
    # Setup data module
    logger.info("Load data module, prepare- and setup data...")
    cfg['dataset']['dense_data_representation'] = True
    seed_everything(seed=cfg['dataset']['seed'], workers=True)
    data_module = CounterfactualGraphDataModule(**cfg['dataset'])
    data_module.prepare_data()
    data_module.setup()
    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
    # Get dataset name:
    dataset_name = cfg['dataset']['dataset_name']
    # Initialize vae
    path = cfg['predicter'][dataset_name]['vae_checkpoint_path']
    logger.info(f'Load model from: {path}')
    vae = ModelFactory.load_model_from_checkpoint(path, model_name='PEGVAE')
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
    # Number of counterfactual examples to generate:
    n = cfg['predicter']['sample_size']
    # Evaluate:
    output = evaluate_nn_baselines(vae=vae, classifier=classifier, train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device, n=n, **cfg['predicter']['nn_configuration'])
    logger.info("Exiting")
    return output

if __name__ == '__main__':
    main()
