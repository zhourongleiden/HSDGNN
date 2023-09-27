import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('current_dir: {}'.format(file_dir))
sys.path.append(file_dir)
import torch
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.HSDGNN import HSDGNN as Network
from model.trainer import Trainer
from lib.utils import init_seed
from lib.dataloader import get_dataloader
import warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(3)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--mode', default='train', type=str)
args.add_argument('--model', default='HSDGNN', type=str)
################# to be modified ##############
args.add_argument('--dataset', default='PSML', type=str)
################# to be modified ##############
#get basic configuration
args1 = args.parse_args()
config_file = './config_file/HSDGNN_{}.conf'.format(args1.dataset)
config = configparser.ConfigParser()
config.read(config_file)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--steps_per_day', default=config['data']['steps_per_day'], type=int)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
#train
args.add_argument('--device', default=config['train']['device'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

#print
args = args.parse_args()
print(args)

#init seed
init_seed(args.seed)

#init device
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args)

#init model
model = Network(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

#print parameters
total_num = sum([param.nelement() for param in model.parameters()])
print('Total params num: {}'.format(total_num))

#init loss function, optimizer
loss = torch.nn.L1Loss().to(args.device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_scheduler= torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load(os.path.join(current_dir,'path_to_the_best_model.pth')))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError