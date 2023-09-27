# HSDGNN
This is the implementation of our paper entitled "Graph Neural Networks for Multivariate Time-Series Forecasting via Hierarchically Spatiotemporal Dependencies Learning".
## Contents
### folder
* config_file: the general parameter configuration of our model on different datasets <br>  
* data: the datasets used in our experiment <br>
* experiment: to store the model after training <br>
* lib: scripts related to data processing, evaluation metrics, and initialization <br>
* model: implementation of the HSDGNN model <br>
### File
* main.py: the script for training <br>
* environment.yml: basic environment configuration
## Environment
* Install conda environment from .yml file  
`conda env create --file environment.yml`
## Model training
Run `python main.py --dataset dataset_name` to train a HSDGNN model from scratch <br>
(Please choose the "dataset_name" from {PEMSD4, PEMSD5, PEMSD8, PEMSD11, PSML})

