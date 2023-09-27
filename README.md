# HSDGNN
This is the implementation of our paper entitled "Graph Neural Networks for Multivariate Time-Series Forecasting via Learning Hierarchical Spatiotemporal Dependencies".
![image text](https://github.com/zhourongleiden/HSDGNN/blob/main/Framework.png)
Figure: Overview of the proposed HSDGNN model. Herein, the model consists of five modules: Intra-dependency Learning, Temporal-dependency Learning, Dynamic Topology Generation,Spatio-dependency learning, and Output Module.
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
## Data
Please download the data for experiment from [Google Drive](https://drive.google.com/file/d/1qoGP0L3ua4ZAwLf_jeBNJoBo69pqAmvV/view?usp=share_link), 
unzip the file to create the "data" folder.
## Environment
* Please install conda environment from .yml file  
`conda env create --file environment.yml`
## Model training
Plese run `python main.py --dataset dataset_name` to train a HSDGNN model from scratch <br>
(Please choose the "dataset_name" from {PEMSD4, PEMSD5, PEMSD8, PEMSD11, PSML})

