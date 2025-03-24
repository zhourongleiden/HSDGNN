# HSDGNN
This is the implementation of our paper entitled "Graph Neural Networks for Multivariate Time-Series Forecasting via Learning Hierarchical Spatiotemporal Dependencies".  <br>
![image text](https://github.com/zhourongleiden/HSDGNN/blob/main/Framework.png)  <br>
Figure: Overview of the proposed HSDGNN model. Herein, the model consists of five modules: Intra-dependency Learning, Temporal-dependency Learning, Dynamic Topology Generation, Spatio-dependency Learning, and Output Module.
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
Please download the data for the experiment from [Google Drive](https://drive.google.com/file/d/1p_b8y9Cl2v-hZYPTqVmV9YvTq9YqvYe1/view?usp=share_link), 
unzip the file to create the "data" folder.
## Environment
* Please install the conda environment from the .yml file  
`conda env create --file environment.yml`
## Model training
Please run `python main.py --dataset dataset_name` to train a HSDGNN model from scratch <br>
(Please choose the "dataset_name" from {PEMSD4, PEMSD5, PEMSD8, PEMSD11, PSML, NSRDB})
## Citation
If you find this repository useful, please cite:<br>
@article{zhou2025graph,<br>
  title={Graph Neural Networks for multivariate time-series forecasting via learning hierarchical spatiotemporal dependencies},<br>
  author={Zhou, Zhou and Basker, Ronisha and Yeung, Dit-Yan},<br>
  journal={Engineering Applications of Artificial Intelligence},<br>
  volume={147},<br>
  pages={110304},<br>
  year={2025},<br>
  publisher={Elsevier}<br>
}
