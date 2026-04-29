import logging

logger = logging.getLogger(__name__)

from counterfactual_graph_generation.data.make_dataset import CounterfactualGraphDataModule
from matplotlib import pyplot as plt
from collections import Counter
from omegaconf import OmegaConf

### A notebook with the purpose of visulizing statistics on the preprocessed graphs
def get_data_module(cfg):
    print("Load data module, prepare- and setup data...")
    data_module = CounterfactualGraphDataModule(**cfg)
    data_module.prepare_data()
    data_module.setup()
    return data_module

def log_dm_info(dm):
    # Size of dataset:
    print("Train dataset size: ", len(dm.data_train))
    print("Validation dataset size: ", len(dm.data_val))
    print("Test dataset size: ", len(dm.data_test))
    print("Total size: ",  len(dm.dataset))

    # Find number of node attributes
    print("Number of node attributes: ", dm.dataset[0].x.shape[1])
    has_edge_attributes = dm.dataset[0].edge_attr != None
    print("Has edge attributes? ", has_edge_attributes)
    if has_edge_attributes:
        print("Number of edge attributes: ", dm.dataset[0].edge_attr.shape[1])

    # Find largest graph in dataset
    max_num_nodes = 0
    num_nodes = []
    all_labels = []
    node_label_frequencies = None
    for _, graph in enumerate(dm.dataset):
        max_num_nodes = max(graph.x.shape[0], max_num_nodes)
        num_nodes.append(graph.x.shape[0])
        all_labels.append(graph.y.item())
        if node_label_frequencies != None:
            node_label_frequencies += graph.x.sum(dim=0)
        else:
            node_label_frequencies = graph.x.sum(dim=0)
    return num_nodes, all_labels, node_label_frequencies

def main(plotting=False):
    paths = ['./config/dataset/aids.yaml', './config/dataset/mutagenicity.yaml', './config/dataset/nci1.yaml']
    for path in paths:
        config = OmegaConf.load(path)
        # Stup data module
        dm = get_data_module(config)
        dm.setup()
        # Log dataset info
        print(f'logging dataset statistics for: {path}')
        num_nodes, all_labels, node_label_frequencies = log_dm_info(dm)
        if plotting:
            # Plot distribution
            title = "Node distribution in:" + config['dataset_name'] + ". Largest graph: " + str(max(num_nodes))
            xlabel = "Value"
            ylabel = "Frequency"
            plt.hist(num_nodes, bins=15, edgecolor='black')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

            # Show the distribution between different classes
            counter = Counter(all_labels)
            labels, frequencies = zip(*counter.items())
            title = "Class label occurence"
            xlabel = "Class"
            ylabel = "Frequency"
            plt.bar(labels, frequencies, edgecolor='black')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

            # Node label distribution:
            title = "Node class label occurence"
            xlabel = "Node class"
            ylabel = "Frequency"
            plt.bar(range(len(node_label_frequencies)), node_label_frequencies, edgecolor='black')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
    return

if __name__ == '__main__':
    main()
