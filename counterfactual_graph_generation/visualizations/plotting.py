import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import os
from PIL import Image
import random

from rdkit import Chem
from enum import Enum
from io import BytesIO
from rdkit.Chem.Draw.rdMolDraw2D import PrepareMolForDrawing, MolDraw2DCairo


from counterfactual_graph_generation.utils import remove_isolated_nodes_from_nx_graph

def plot_roc_curve(fpr, tpr, thresholds, roc_auc=None, idx=None):
    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if idx != None:
        optimal_threshold = thresholds[idx]
        plt.scatter(fpr[idx], tpr[idx], c='blue', marker='x', s=100, label='Optimal Threshold: {:.2f}'.format(optimal_threshold))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return plt

def plot_getometric_mean(geometric_mean, thresholds, idx):
    plt.figure()
    lw = 2
    plt.plot(thresholds, geometric_mean, color='green', lw=lw, label='Geometric Mean')
    plt.axvline(x=thresholds[idx], color='red', linestyle='--', label='Optimal Threshold: {:.2f}'.format(thresholds[idx]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Threshold')
    plt.ylabel('Geometric Mean')
    plt.title('Geometric Mean as a Function of Thresholds')
    plt.legend(loc="lower right")
    return plt

def plot_nxgraphs(G_input_nx, G_recon_nx, G_nx,
                  include_reconstruction=True, show=True,
                  title=None, figsize=(15,5),
                  remove_isolated_nodes=True
    ):

    num_subplots = 3 if include_reconstruction else 2
    fig, axs = plt.subplots(1,num_subplots)

    if remove_isolated_nodes:
        remove_isolated_nodes_from_nx_graph(G_input_nx)
        remove_isolated_nodes_from_nx_graph(G_recon_nx)
        remove_isolated_nodes_from_nx_graph(G_nx)

    # Nx positions
    pos = nx.kamada_kawai_layout(G_input_nx)

    # Plot input graph
    nx.draw(G_input_nx, pos=pos, ax=axs[0])
    axs[0].set_title('Input Graph')

    # Plot CF graph
    nx.draw(G_nx, ax=axs[1])
    axs[1].set_title('Counterfactual Graph')

    # Plot reconstruction
    if include_reconstruction:
        nx.draw(G_recon_nx, ax=axs[2])
        axs[2].set_title('Reconstructed Graph')

    if show:
        plt.show()
    return fig

def sample_nxgraphs_from_lists(list_of_input_graphs, list_of_recon_graphs, list_of_cf_graphs,
                          num_samples=10, seed=None
    ):
    if len(list_of_input_graphs) != len(list_of_recon_graphs) \
        or len(list_of_input_graphs) != len(list_of_cf_graphs):
        raise ValueError("All lists must have the same length.")
    if seed is not None:
        random.seed(seed)
    triplets, indexes = [], []
    for _ in range(num_samples):
        index = random.randint(0, len(list_of_input_graphs)- 1)
        input_graph, reconstruction_graph, counterfactual_graph \
              = list_of_input_graphs[index], list_of_recon_graphs[index], list_of_cf_graphs[index]
        triplets.append((input_graph, reconstruction_graph, counterfactual_graph))
        indexes.append(index)
    return triplets, indexes

def plot_nxgraphs_to_pdf(triplets, indexes,
                         figsize=(15,5), include_reconstruction=True,
                         pdf_file="example_graphs.pdf"
    ):
    os.makedirs('./data/visualizations/', exist_ok = True)
    with PdfPages(pdf_file) as pdf:
        for _, (triplet, index) in enumerate(zip(triplets, indexes)):
            fig = plot_nxgraphs(triplet[0], triplet[1], triplet[2], title=str(index),
                                figsize=figsize, include_reconstruction=include_reconstruction,
                                show=False
            )
            pdf.savefig(fig)
            plt.close(fig)

def prepare_mol(mol):
    try:
        mol_draw = PrepareMolForDrawing(mol)
    except Chem.KekulizeException:
        print(Chem.KekulizeException)
        mol_draw = PrepareMolForDrawing(mol, kekulize=False)
        Chem.SanitizeMol(mol_draw, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    return mol_draw

def plot_mol(mol, legend='', highlightAtoms=[], show=False):
    d2d = MolDraw2DCairo(350,300)
    d2d.DrawMolecule(mol, legend=legend, highlightAtoms=highlightAtoms)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    img = Image.open(bio)
    if show:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    return img

def plot_compare_mol(imgs, index, method, dataset,
                     buffer=5, show=True, save=False,
                     figsize=(15,5)
    ):
    height, width = 0, 0
    for img in imgs:
        height = max(height,img.height)
        width += img.width
    width += buffer*(len(imgs)-1)
    res = Image.new("RGBA", (width,height))
    x = 0
    for img in imgs:
        res.paste(img,(x, 0))
        x += img.width + buffer
    plt.figure(figsize=figsize)
    plt.imshow(res)
    plt.axis('off')
    if save:
        os.makedirs(f'./data/visualizations/{dataset}', exist_ok = True)
        plt.savefig(f'./data/visualizations/{dataset}/{method}_{index}.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return res

def pyg_to_mol_aids(pyg_mol):
    mol = Chem.RWMol()
    X = pyg_mol.x.numpy().tolist()
    X = [Chem.Atom(x_map_to_aids(x.index(1)).name)for x in X]
    E = pyg_mol.edge_index.t()
    for x in X: mol.AddAtom(x)
    for (u, v), attr in zip(E, pyg_mol.edge_attr):
        u, v = u.item(), v.item()
        attr = attr.numpy().tolist()
        attr = e_map_to_aids(attr.index(1), reverse=True)
        if mol.GetBondBetweenAtoms(u, v):
            continue
        mol.AddBond(u, v, attr)
    return mol

def e_map_to_aids(bond_type, reverse=False):
    if not reverse:
        if bond_type == Chem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.BondType.TRIPLE:
            return 2
        else:
            raise Exception("No bond type found")

    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")

class x_map_to_aids(Enum):
    C = 0
    O = 1
    N = 2
    Cl = 3
    F = 4
    S = 5
    P = 6
    Na = 7
    Br = 8
