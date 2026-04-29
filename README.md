# counterfactual_graph_generation

A project for generating graph counterfactuals.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── counterfactual_graph_generation  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

# Setup project:

To setup this project first create and activate conda environment:

```
conda create -n cfg python=3.10
conda activate cfg
```

Navigate to the project repository. For development install:

```
make dev_requirements
```

Otherwise use:
```
make requirements
```

To prepare dataset for training use for instance:
```
make data dataset='aids'
```

To train, get predictions, and evalaute a specific model use the format:
```
make train dataset='aids' model='AidsClassifier'
make predict dataset='aids'
make evaluate
```
Note that 'make predict' and 'make evaluate' relies heavily on the configurations in "predicter_config.yaml" and "evaluation_config.yaml" respectively.

Also note, that if no arguments are granted to 'make' then the default configuration values will be applied.

### Note:
rdkit requires libXrender to be installed to work properly. If pip is used for the installation, then this might not be the case, and libXrender should be installed seperately. This can be alleviated by using conda-forge instead of pip for the installation of rdkit.

# Wandb
To login to wandb do and enable cloud syncing your experiments:
```
wandb login
```
To initialise a parameters sweep use:
```
wandb sweep config/sweeps/aids-sweep.yaml
```
To start a sweep agent use:
```
wandb agent aah/counterfactual_graph_generation/<sweep-id>
```
To stop the sweep use:
```
wandb sweep --stop aah/counterfactual_graph_generation/<sweep-id>
```

The wandb artifact cache can become quite bloated after some time. The following command prunes the cache to delete files which haven't been used lately:
```
wandb artifact cache cleanup 1GB
```

# Usefull commands
Display folder sizes from terminal:
```
du -sh */
```

# StochMan
For the implementation of counterfactual graph generation we employ the StochMan library: https://github.com/MachineLearningLifeScience/stochman


# Connect to a Jupyter-notebook server:
Running jupyter-notebooks on the cluster can be done by:

```
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

A Jupyter notebook server will then be created. To connect to this server a VS Code extension might be needed (e.g. JupyterHub). Having installed this one can connect to the jupyter-server by opening the notebook in question, selecting "select kernel", and then input the URL of the Jupyter-server.

# Pipeline:
The full training pipeline runs like so:
```
make train dataset=aids model=AidsClassifierDense
make train dataset=aids model=AidsPegvae
```
After training has completed, the model can be picked based on either Validation loss or reconstruction loss (validation). The path to the selected models should be added to the "predition" configuration file. The configuration file also contains whether prediction should be done on the train, validation or test-set, as well as the hyperparamters of the baseline methods. When this has been done do:
```
make predict dataset=aids
```
The predictions will now be saved in data/predictions/val_aids.pt. Now, you can proceed to evaluation. Now the evaluation can be done by running:
```
make evaluate path=data/predictions/val_aids.pt
```
Here, the path variable containes the path to the predictions. The evaluation will produce the following:
- Dataframes containing the method statistics saved to data/predictions/<dataset>_method_statistcs. There will be one dataframe for each method aswell as one for the mean value of all methods.
- Dataframes for Anna for the visualisation of graph property distributions.
- Visualizations with example graphs and molecules. Theese are saved to data/visualizations/val_aids and data/visualizations.
Now, we can produce validity-fidelity graphs. These are created by running:
```
python counterfactual_graph_generation/visualizations/visualize_method_statistics.py --prefix=val_nci1 --exclude=10 --normalize=True
python counterfactual_graph_generation/visualizations/visualize_method_statistics.py --prefix=val_mutagenicity --exclude=0 --normalize=False
```
Also, Latex tables of the evaluation results can be produced thorugh the Jupyter Notebook notebooks/visualize_latex_tables.ipynb.

Lastly, the notebook notebooks/visualization_factual_counterfactual.ipynb produces plots for each of the node property distributions.
