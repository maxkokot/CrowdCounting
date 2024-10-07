Crowd Counting
==============================

Crowd Counting using Xception. The repository presents a solution for Crowd Counting problem evaluated on Mall Dataset. Here we use pretraining on Shanghai Tech Dataset according to method mentioned in the book (Aurélien Géron. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. ” O’Reilly Media, Inc.”, 2022). As the starting point, we use Xception model pretrained on Imagenet Dataset.

This code is my contribution for following project: https://github.com/Di40/CrowdCounting_Xception_CSRNet/tree/main

## Usage

### Downloading data

First ensure that kaggle package has been installed on your system. Also generate your API token and put at at ~/.kaggle/kaggle.json. The procedure is described here: https://www.kaggle.com/docs/api
Next download datasets by using src/download_data.sh

```
bash download_data.sh
```


### Fit Xception and evaluate scores

Run training by using src/train.py

```
python train.py --model_config_path ../configs/model_config.yaml
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── data                    <- Datasets for training.
    │
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Trained and serialized models.
    │
    ├── notebooks               <- Jupyter notebooks.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Project report.
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Tools for data manipulation
    │   │   └── data.py
    │   │
    │   ├── models              <- Tools for modeling
    │   │   └── model.py
    │   │
    │   ├── unzip.py            <- Script to unzip data to data folder
    │   │
    │   ├── download_data.sh    <- Script to download the datasets
    │   │
    │   └── train.py            <- Script to train the models
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
