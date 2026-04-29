import logging

logger = logging.getLogger(__name__)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
from torch_geometric.utils import to_networkx
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from counterfactual_graph_generation.utils import dense_graph_to_data_object, batched_dense_graph_to_batched_data_object, save_df_as_csv
from counterfactual_graph_generation.metrics.ood_stat import eval_graph_list
from counterfactual_graph_generation.visualizations.plotting import *
from counterfactual_graph_generation.visualizations.plotting import prepare_mol

def maximal_shortest_path_length(G):
    length = dict(nx.all_pairs_shortest_path_length(G))
    shortest_lenghts = [length[source][node] for source in G.nodes for node in G.nodes if source != node]
    return np.max(shortest_lenghts)

def compute_graph_metrics(G):
    den = nx.density(G)
    ncc = nx.number_connected_components(G)
    avgsp = nx.average_shortest_path_length(G) if nx.is_connected(G) else None
    maxsp = maximal_shortest_path_length(G) if nx.is_connected(G) else None
    return {'density': den, 'connected compenents': ncc, 'max-shortest path': maxsp}

def absolute_confidence_difference(probit_f, probit_cf, input_class): # Only valid for binary classification
    desired_class = 1 - input_class
    f_dist_to_desired = torch.abs(desired_class - probit_f[1])
    cf_dist_to_desired = torch.abs(desired_class - probit_cf[1])
    diff_in_distance = f_dist_to_desired - cf_dist_to_desired # Distance to the desired should decrease for the counterfactual
    return diff_in_distance.item()

def flip_ratio(target, y_cf, debug=False): # Only valid for binary classification
    flipped = (y_cf == target).sum().item()
    logger.debug(f'Flipped: {flipped}')
    logger.debug(f'Number of data points: {target.shape[0]}')
    logger.debug(f'Ratio: {flipped / target.shape[0]}')
    return flipped / target.shape[0]

def latent_distance(latent_f, latent_cf, type='euclidian'):
    if type == 'euclidian':
        dist = np.linalg.norm(latent_f - latent_cf, ord=2)
    elif type == 'cosine':
        dist = cosine_similarity(latent_f, latent_cf).item()
    return dist

def calculate_MMD(input_graphs, pred_graphs, methods=["degree"]):
    mmd = eval_graph_list(input_graphs, pred_graphs, methods=methods)
    return mmd

def dense_graph_to_networkx(G_input_dense):
    G_input_pyg = dense_graph_to_data_object(**G_input_dense)
    G_input_nx = to_networkx(data=G_input_pyg, to_undirected=True)
    return G_input_nx

def dict_to_list(dict_of_lists):
    return [dict(zip(dict_of_lists,t)) for t in zip(*dict_of_lists.values())]

def list_to_dict(list_of_dicts):
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

@hydra.main(config_path="../config", config_name="config.yaml", version_base="1.2")
def main(cfg):
    for path in cfg['evaluation']['evaluation_paths']:
        # load results
        file_prefix = path.split("/")[-1][:-3]
        logger.info(f'------- Loading evaluation data from {file_prefix} -------')
        data = torch.load(path, map_location=torch.device('cpu'))

        ######### Classifier evaluation: ######
        # ( Done during training )

        ###### Evaluation of VAE #######
        # KL-loss: Read from model evaluation
        # Reconstruction loss: Read from model evaluation
        # ELBO: Read from model evaluation
        # Accuracy: See below
        # F1: See below

        y_input = data['input']['y']
        y_reconstructed = data['reconstructions']['y']
        acc_recon = balanced_accuracy_score(y_input, y_reconstructed)
        f1_recon = f1_score(y_input, y_reconstructed)

        logger.info("------- Evaluating VAE reconstructions -------")
        logger.info(f'Accuracy: {acc_recon}')
        logger.info(f'F1: {f1_recon}\n')


        ###### Produce classification tables for Anna ######
        path = f'./data/predictions/{file_prefix}_distribution_evaluation/'

        cf_dict = data['counterfactuals']['main_method'][0]
        f_dict = data['input']
        y_pred_counterfactual = cf_dict['y'].view(-1) # PREDICTED class of generated COUNTERFACTUAL
        y_target = 1 - f_dict['y'].view(-1) # TARGET class of generated COUNTERFACTUAL
        y_pred = f_dict['probits'].argmax(dim=1).view(-1) # PREDICTED class of FACTUAL graph
        y_input = f_dict['y'].view(-1) # TARGET class if FACTUAL graph

        # Node counts
        # CF
        counterfactual_number_of_nodes = cf_dict['B'].sum(dim=2).view(-1)
        df = pd.DataFrame({"Number_of_nodes": counterfactual_number_of_nodes, "prediction": y_pred_counterfactual, "target": y_target})
        save_df_as_csv(df, path + f'{file_prefix}_counterfactual_node_count.csv')
        # True
        number_of_nodes = f_dict['B'].sum(dim=2).view(-1)
        df = pd.DataFrame({"Number_of_nodes": number_of_nodes, "prediction": y_pred, "target": y_input})
        save_df_as_csv(df, path + f'{file_prefix}_factual_node_count.csv')

        # Edge counts
        counterfactual_number_of_edges= 0.5 * cf_dict['A'].sum(dim=(2,3)).view(-1)
        df = pd.DataFrame({"Number_of_edges": counterfactual_number_of_edges, "prediction": y_pred_counterfactual, "target": y_target})
        save_df_as_csv(df, path + f'{file_prefix}_counterfactual_edge_count.csv')

        number_of_edges= 0.5 * f_dict['A'].sum(dim=(2,3)).view(-1)
        df = pd.DataFrame({"Number_of_edges": number_of_edges, "prediction": y_pred, "target": y_input})
        save_df_as_csv(df, path + f'{file_prefix}_factual_edge_count.csv')

        # Densities:
        # CF
        nodes_densities_of_cfs = (2 * counterfactual_number_of_edges) / (counterfactual_number_of_nodes * (counterfactual_number_of_nodes - 1))
        df = pd.DataFrame({"Densities": nodes_densities_of_cfs, "prediction": y_pred_counterfactual, "target": y_target})
        save_df_as_csv(df, path + f'{file_prefix}_counterfactual_densities.csv')
        # True
        nodes_densities_of_factuals = (2 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1))
        df = pd.DataFrame({"Densities": nodes_densities_of_factuals, "prediction": y_pred, "target": y_input})
        save_df_as_csv(df, path + f'{file_prefix}_factual_densities.csv')

        # Node distributions
        # CF
        F_cf = cf_dict['F']
        F_cf_node_distribution = F_cf.sum(dim=2)
        cf_not_a_node = (1-F_cf.sum(dim=1)).sum(dim=1, keepdim=True)
        F_cf_node_distribution = torch.cat([F_cf_node_distribution, cf_not_a_node], dim=1) / F_cf.shape[-1]
        shannon_entropy_cf = (-F_cf_node_distribution*(torch.log(F_cf_node_distribution + 1.0e-07))).sum(dim=1)
        df = pd.DataFrame({"Shannon Entropy": shannon_entropy_cf, "prediction": y_pred_counterfactual, "target": y_target})
        save_df_as_csv(df, path + f'{file_prefix}_counterfactual_shannon_entropy.csv')
        # True
        F_in = f_dict['F']
        F_in_node_distribution = F_in.sum(dim=2)
        in_not_a_node = (1-F_in.sum(dim=1)).sum(dim=1, keepdim=True)
        F_true_node_distribution = torch.cat([F_in_node_distribution, in_not_a_node], dim=1) / F_in.shape[-1]
        shannon_entropy_true = (-F_true_node_distribution*(torch.log(F_true_node_distribution + 1.0e-07))).sum(dim=1)
        df = pd.DataFrame({"Shannon Entropy": shannon_entropy_true, "prediction": y_pred, "target": y_input})
        save_df_as_csv(df, path + f'{file_prefix}_factual_shannon_entropy.csv')

        ###### Evaluation of generated counterfactuals ######
        # - Cosine similarty
        # - Graph edit distance
        # - MAD
        # - Latent space distance
        # - Graph statistics

        # Rewrite input graphs:
        list_of_input_graphs = dict_to_list(data['input'])
        list_of_recon_graphs = dict_to_list(data['reconstructions'])
        list_of_nx_graphs_input = [dense_graph_to_networkx(graph) for graph in list_of_input_graphs]
        list_of_nx_graphs_recon = [dense_graph_to_networkx(graph) for graph in list_of_recon_graphs]

        # Graph Edit Distance and graphs statistics
        logger.info("------- Baselines reconstructions -------")
        methods_to_evaluate = data['counterfactuals'].keys() #['random_sampled_CFs_list', 'main_method']
        summary_dict_mean = {}
        summary_dict_std = {}
        for _, method in enumerate(methods_to_evaluate):
            logger.info(f'Evaluating: {method}')
            # Prepare data
            cfs = data['counterfactuals'][method][0] # Only pick the first counterfactual for now
            if 'train_info' in cfs: # Unpredictable behaviour if this is not done
                del cfs['train_info']
            list_of_baseline_graphs = dict_to_list(cfs)
            list_of_nx_graphs_baseline = [dense_graph_to_networkx(graph) for graph in list_of_baseline_graphs]
            # Evaluate graph edit distance and graph statistics using nx graphs
            results_list_of_dictionaries = []
            for i, G_dense in enumerate(list_of_baseline_graphs):
                results_dictionary = {}
                # Graph representations
                G_nx = list_of_nx_graphs_baseline[i]
                G_input_dense = list_of_input_graphs[i]
                G_recon_dense = list_of_recon_graphs[i]
                G_input_nx = list_of_nx_graphs_input[i]

                # plot molecules
                if method == 'main_method' and file_prefix.split('_')[-1]=='aids':
                    pyg_mol_input = dense_graph_to_data_object(**G_input_dense)
                    mol_input = prepare_mol(pyg_to_mol_aids(pyg_mol_input))
                    pyg_mol_recon = dense_graph_to_data_object(**G_recon_dense)
                    mol_recon = prepare_mol(pyg_to_mol_aids(pyg_mol_recon))
                    pyg_mol = dense_graph_to_data_object(**G_dense)
                    mol = prepare_mol(pyg_to_mol_aids(pyg_mol))
                    imgs = []
                    imgs.append(plot_mol(mol_input, legend='Input'))
                    imgs.append(plot_mol(mol_recon, legend='Reconstruction'))
                    imgs.append(plot_mol(mol, legend='Counterfactual'))
                    res = plot_compare_mol(imgs, index=i, method=method, dataset=path.split('/')[-1].split('.')[0], save=True, show=False)

                # Compute graph statistics
                metrics = compute_graph_metrics(G_nx)
                results_dictionary.update(metrics)
                # Calculate graph distances
                results_dictionary['GED (Fidelity)'] = nx.graph_edit_distance(G_input_nx, G_nx, timeout=0.01)
                results_dictionary['Absolute Difference (Validity)'] = absolute_confidence_difference(G_input_dense['probits'], G_dense['probits'], G_input_dense['y'])
                results_dictionary['Latent Cossimilary (Fidelity)'] = latent_distance(G_input_dense['graph_embedding'].reshape(1, -1),
                                                                    G_dense['graph_embedding'].reshape(1, -1), type='cosine')
                results_dictionary['Latent Euclidean Distance (Fidelity)'] = latent_distance(G_input_dense['mu'],  G_dense['z'], type='euclidian')
                results_dictionary['Target Class'] = y_target.reshape(-1)[i].item()
                results_dictionary['Predicted Counterfactual Class'] = cfs['y'].reshape(-1)[i].item()
                results_list_of_dictionaries.append(results_dictionary)

            # Calculate MMD
            logger.info('Calculating MMD')
            mmd_degree = calculate_MMD(list_of_nx_graphs_input, list_of_nx_graphs_baseline, methods=["degree"])["degree"]
            # Calculate flip ratio
            fr = flip_ratio(y_target.reshape(-1), cfs['y'].reshape(-1))
            # Plotting
            if cfg['evaluation']['plot']:
                example_triplets, example_indexes = sample_nxgraphs_from_lists(
                    list_of_nx_graphs_input, list_of_nx_graphs_recon, list_of_nx_graphs_baseline,
                    num_samples=10, seed=42
                )
                plot_nxgraphs_to_pdf(example_triplets, example_indexes,
                                        include_reconstruction=True,
                                        pdf_file=f"./data/visualizations/example_graphs_{method}.pdf"
                )
            # Create dataframe with results
            df = pd.DataFrame(results_list_of_dictionaries)
            save_df_as_csv(df, f'./data/predictions/{file_prefix}_method_statistics/' + f'{method}_counterfactual_statistics.csv', )
            # Summarize overall statistics
            summary_dict_mean[f'{method} (mean)'] = df.mean()
            summary_dict_mean[f'{method} (mean)']['flip-ratio (Validity)'] = fr
            summary_dict_mean[f'{method} (mean)']['MMD Degree (Fidelity)'] = mmd_degree
            summary_dict_std[f'{method} (mean)'] = df.std()
            logger.debug(f'Head of resulting dataframe: {df.head()}')
            logger.info(f'Exiting \n ')
        summary_df_mean = pd.DataFrame(summary_dict_mean)
        summary_df_std = pd.DataFrame(summary_dict_std)
        save_df_as_csv(summary_df_mean, f'./data/predictions/{file_prefix}_method_statistics/' + f'aggregated_mean_counterfactual_statistics.csv')
        save_df_as_csv(summary_df_std, f'./data/predictions/{file_prefix}_method_statistics/' + f'aggregated_std_counterfactual_statistics.csv')
        logger.info(f'Summary of means: \n {summary_df_mean.round(decimals=4)}')
        logger.info(f'Summary of stds: \n {summary_df_std.round(decimals=4)}')

if __name__ == '__main__':
    main()
