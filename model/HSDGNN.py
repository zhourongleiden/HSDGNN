import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# clean version
class HSDGNN_block(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, embed_dim):
        super(HSDGNN_block, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.gru1 = nn.GRU(embed_dim, rnn_units)
        self.gru2 = nn.GRU(rnn_units, rnn_units)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, rnn_units, rnn_units))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, rnn_units))
        self.diff = nn.Conv2d(rnn_units*2, rnn_units, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True) 
        self.dropout = nn.Dropout(p=0.1)
        self.x_embedding = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(1, 16)),
                        ('sigmoid1', nn.Sigmoid()),
                        ('fc2', nn.Linear(16, 2)),
                        ('sigmoid2', nn.Sigmoid()),
                        ('fc3', nn.Linear(2, embed_dim))]))
        self.fc1 = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(input_dim, 16)), 
                        ('sigmoid1', nn.Sigmoid()),
                        ('fc2', nn.Linear(16, 2)),
                        ('sigmoid2', nn.Sigmoid()),
                        ('fc3', nn.Linear(2, embed_dim))]))    
        self.fc2=nn.Sequential(
        OrderedDict([('fc1', nn.Linear(rnn_units, 16)),
                        ('sigmoid1', nn.Sigmoid()),
                        ('fc2', nn.Linear(16, 2)),
                        ('sigmoid2', nn.Sigmoid()),
                        ('fc3', nn.Linear(2, embed_dim))]))     

    def dynamic_dependency(self, output_of_gru, node_embeddings_all):
        #shape of output_of_gru: (T, B, N, rnn_units)
        #shape of node_embeddings_all:[(T, B, N, embed_dim),(N, embed_dim)]
        filter = self.fc2(output_of_gru) # (T, B, N, embed_dim)    
        nodevec = torch.tanh(torch.mul(node_embeddings_all[0], filter))  # (T, B, N, embed_dim) 
        supports1 = torch.stack([torch.eye(self.num_nodes)]*output_of_gru.shape[0]).to(node_embeddings_all[0].device) # (T, N, N)
        supports2 = F.relu(torch.matmul(nodevec, nodevec.permute(0,1,3,2))) # (T, B, N, N) 
        x_g1 = torch.einsum("tnm,tbmc->tbnc", supports1, output_of_gru) #(T, B, N, rnn_units)
        x_g2 = torch.einsum("tbnm,tbmc->tbnc", supports2, output_of_gru) 
        x_g = x_g1 + x_g2    
        weights = torch.einsum('nd,dcr->ncr', node_embeddings_all[1], self.weights_pool)  
        bias = torch.matmul(node_embeddings_all[1], self.bias_pool)  
        x_g = torch.einsum('tbnc,ncr->tbnr', x_g, weights) + bias  #T, B, N, rnn_units
        return x_g

    def forward(self, x, node_embeddings_all):
        #shape of x: (B, T, N, input_dim)
        #shape of node_embeddings_all:[(T, B, N, embed_dim),(N, embed_dim)]
        #shape of output: (B, T, N, rnn_units)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        B, T, N = x.shape[0], x.shape[1], x.shape[2]   
        #GRU for each time series  
        node_embed = self.x_embedding(x.unsqueeze(-1)) # (B, T, N, input_dim, embed_dim)
        supports_node1 = torch.eye(self.input_dim).to(node_embeddings_all[0].device)
        supports_node2 = F.relu(torch.matmul(node_embed, node_embed.permute(0,1,2,4,3))) # (B, T, N, input_dim, input_dim)
        x1 = torch.einsum("ji,btni->btnj", supports_node1, x)
        x2 = torch.einsum("btnji,btni->btnj", supports_node2, x) #(B, T, N, input_dim) 
        input_for_fc1 = x1 + x2 
        input_for_gru1 = self.fc1(input_for_fc1).permute(1,0,2,3)
        h0 = torch.zeros(1, B*N, self.rnn_units).to(input_for_gru1.device)
        output_of_gru1, _ = self.gru1(input_for_gru1.reshape(T,B*N, self.embed_dim), h0) 
        #GCN for each time step
        input_for_gru2 = []
        output_of_gru1 = output_of_gru1.reshape(T,B,N,self.rnn_units)
        input_for_gru2.append(output_of_gru1) 
        diff1 = self.dynamic_dependency(output_of_gru1, node_embeddings_all)
        input_for_gru2.append(diff1)
        input_for_gru2 = self.diff(torch.cat(input_for_gru2, dim=3).permute(1,3,2,0)).permute(3,0,2,1)
        input_for_gru2 = self.dropout(input_for_gru2)
        #GRU2 for modeling the change of dependencies
        h0 = torch.zeros(1, B*N, self.rnn_units).to(input_for_gru2.device)
        output_of_gru2,_ = self.gru2(input_for_gru2.reshape(T,B*N, self.rnn_units), h0) 
        return output_of_gru2.reshape(T,B,N,self.rnn_units).permute(1,0,2,3)


class HSDGNN(nn.Module):
    def __init__(self, args):
        super(HSDGNN, self).__init__()
        self.batch_size = args.batch_size
        self.rnn_units = args.rnn_units
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.steps_per_day = args.steps_per_day
        self.node_embeddings = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(args.steps_per_day, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))
        self.encoder1 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)   
        self.encoder2 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)       
        self.end_conv1 = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv1_b = nn.Conv2d(1, args.lag, kernel_size=(1, args.rnn_units), bias=True) 
        self.end_conv2 = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)
        self.dropout3 = nn.Dropout(p=0.1)
        self.encoder3 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)       
        self.end_conv2_b = nn.Conv2d(1, args.lag, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)      

    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #read input
        source1 = source[..., 0:-2] 
        node_embedding = self.node_embeddings
        #add time information
        t_i_d_data = source[..., -2]
        T_i_D_emb = self.T_i_D_emb[(t_i_d_data * self.steps_per_day).type(torch.LongTensor)]
        node_embedding = torch.mul(node_embedding, T_i_D_emb)
        d_i_w_data = source[..., -1]
        D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
        node_embedding = torch.mul(node_embedding, D_i_W_emb)
        node_embeddings_all=[node_embedding.permute(1,0,2,3), self.node_embeddings]
        ################
        #Block 1
        output_1 = self.encoder1(source1, node_embeddings_all)     
        output_1 = self.dropout1(output_1[:, -1:, :, :])    
        #CNN based predictor
        output1 = self.end_conv1(output_1)                      
        #Residual
        source1_b = self.end_conv1_b(output_1)
        source2 = source1 - source1_b
        #Block 2
        output_2 = self.encoder2(source2, node_embeddings_all)     
        output_2 = self.dropout2(output_2[:, -1:, :, :])      
        #CNN based predictor
        output2 = self.end_conv2(output_2)
        #Residual
        source2_b = self.end_conv2_b(output_2)
        source3 = source2 - source2_b
        #Block 3
        output_3 = self.encoder3(source3, node_embeddings_all)     
        output_3 = self.dropout3(output_3[:, -1:, :, :])  
        #CNN based predictor         
        output3 = self.end_conv3(output_3)
        return output1 + output2 + output3  