import os
import torch
import numpy as np
import torch.utils.data


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('./data/PEMSD4/PEMSD4.npz')
        data = np.load(data_path)['data']
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PEMSD8/PEMSD8.npz')
        data = np.load(data_path)['data']
    elif dataset == 'PEMSD11':
        data_path = os.path.join('./data/PEMSD11/PEMSD11.npz')
        data = np.load(data_path)['data']
        data = np.transpose(data, (1,0,2))  
    elif dataset == 'PEMSD5':
        data_path = os.path.join('./data/PEMSD5/PEMSD5.npz')
        data = np.load(data_path)['data']
        data = np.transpose(data, (1,0,2))  
    elif dataset == 'PSML':
        data_path = os.path.join('./data/PSML/PSML.npz')
        data = np.load(data_path)['data'][...,0:3]
        data = np.transpose(data, (1,0,2))
    elif dataset == 'NSRDB':
        data_path = os.path.join('./data/NSRDB/NSRDB.npz')
        data = np.load(data_path)['data']
        data[:,:,[0,-2]] = data[:,:,[-2,0]]
        data = np.transpose(data, (1,0,2)) 
        data = data[-data.shape[0]//12:,:,:]  
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        if np.isscalar(self.mean):
            return (data * self.std) + self.mean
        else:
            return (data * self.std[0]) + self.mean[0]

def normalize_dataset(data):
    mean = data.mean(axis=0).mean(axis=0) 
    std = data.reshape(-1,3).std(axis=0) 
    scaler = StandardScaler(mean, std)
    data = scaler.transform(data)
    print('Normalize the dataset by Standard Normalization')      
    return data, scaler

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def Add_Window_Horizon(data, window=3, horizon=1):
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      
    Y = []      
    index = 0
    while index < end_index:
        X.append(data[index:index+window])
        Y.append(data[index+window:index+window+horizon])
        index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(args):
    #load raw st dataset
    data = load_st_dataset(args.dataset)    
    #normalize st data
    data, scaler = normalize_dataset(data)
    #add time info
    _, N, _ = data.shape
    feature_list = [data]
    #time_in_day
    time_ind    = [i%args.steps_per_day / args.steps_per_day for i in range(data.shape[0])]
    time_ind    = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)
    #day_in_week
    day_in_week = [(i // args.steps_per_day)%7 for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(day_in_week)
    #concat
    data = np.concatenate(feature_list, axis=-1)
    #spilit dataset by days or by ratio
    data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio) 
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)  
    #get dataloader
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler
