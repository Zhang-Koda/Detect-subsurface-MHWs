# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 15:31:35 2023

@author: kelis
"""


import numpy as np
import h5py
import scipy.io as scio
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from d2l import torch as d2l


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()  # Clear cuda video memory



"""
# =============================================================================
# #============================function preparation=============================
# =============================================================================
"""


def MSE_cal(Y_hat, y):
    '''
    Calculate the mean squared error between predicted and true values.

    Parameters:
    - Y_hat : Predicted values.
    - y : True values.

    Returns:
    - Mean squared error.
    '''
    temp_data = torch.square(Y_hat - y)
    total = temp_data.numel()
    # Replace NaN values with zeros
    temp_data = torch.nan_to_num(temp_data, nan=0.0)
    mse = temp_data.sum() / total
    
    return mse




def standardization(input_data,comp_data):
    '''
    Standardize input_data using the mean and standard deviation from flattened.

    Parameters:
    - input_data : The data to be standardized.
    - comp_data : The data used to compute mean and standard deviation.

    Returns:
    - np.ndarray: The standardized data.
    '''
    if input_data.size == 0 or comp_data.size == 0:
        raise ValueError("Input arrays must not be empty.")
        
    mean_value = np.nanmean(comp_data_flat)
    std_deviation = np.nanstd(comp_data_flat)

    if std_deviation == 0:
        raise ValueError("Standard deviation is zero. Cannot perform standardization.")

    standardized_data = (input_data - mean_value) / std_deviation
    return standardized_data




def evaluation(net, data_iter, device=None): 
    '''
    Evaluate a neural network using mean squared error on a given dataset.

    Parameters:
    - net : The neural network model.
    - data_iter : The data iterator.
    - device : The device on which to perform computations.

    Returns:
    - The average mean squared error.
    '''
    if isinstance(net, nn.Module):
        net.eval() 
    
    if not device:
        device = next(iter(net.parameters())).device
        
        
    total_mse = 0.0
    num_batches = len(data_iter)
    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            total_mse += MSE_cal(y_hat, y)
            
    average_mse = total_mse / num_batches

    return average_mse
           



def data_generation(net, data_iter, lat, lon, filename):
    '''
    Generate data using a neural network and save it to a (.mat) file.

    Parameters:
    - net : The neural network model.
    - data_iter : The data iterator.
    - lat : Number of latitude points.
    - lon : Number of longitude points.
    - output_file : Path to save the output file.
    - device : The device on which to perform computations.

    Returns:
    - None
    '''
    if isinstance(net, nn.Module):
        net.eval() 
    
    data = np.empty((0,lat,lon))
    
    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):  
            X = X.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            y_hat = y_hat.cpu().numpy()
            data = np.append(data, y_hat,axis=0)

    scio.savemat(filename ,{'data':data}) 









def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)  #variance to remain the same
        
def train_CNN_model(net, train_iter, tst_iter1, tst_iter2, num_epochs, 
               num_lat, num_lon, dataset_trn, dataset_tst1, dataset_tst2):    
    '''
    Train a Convolutional Neural Network (CNN) model.
    
    Parameters:
    - net : The CNN model.
    - train_iter : The training data iterator.
    - tst1_iter : The first test data iterator.
    - tst2_iter : The second test data iterator.
    - num_epochs : Number of training epochs.
    - num_lat : Number of latitude points.
    - num_lon : Number of longitude points.
    - dataset_trn : The training dataset.
    - dataset_tst1 : The 1st test dataset.
    - dataset_tst2 : The 2nd test dataset.
    
    Returns:
    - net : The trained CNN model.
    - Trn_mse : The mse on training data iterator.
    - Tst1_mse : The mse on the first test data iterator.
    - Tst2_mse : The mse on the second test data iterator.
    
    '''
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)

    optimizer = torch.optim.Adadelta(net.parameters())
    
    loss = nn.MSELoss()
    timer = d2l.Timer()

    
    Trn_mse = np.empty(0)
    Tst_mse1 = np.empty(0)
    Tst_mse2 = np.empty(0)
    for epoch in range(num_epochs):
        print('epoch:'+str(epoch))

        net.train()
        trn_mse = 0
        for i, (X, y) in enumerate(train_iter):  
        
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()  
            
            with torch.no_grad():
                mse = MSE_cal(y_hat, y)
    
            timer.stop()
            trn_mse += mse
            
        trn_mse = trn_mse / len(train_iter)
        trn_mse = trn_mse.cpu().numpy()
        Trn_mse = np.append(Trn_mse, trn_mse)
        
        
        tst_mse1 = evaluation(net, tst_iter1)
        tst_mse1 = tst_mse1.cpu().numpy()
        Tst_mse1 = np.append(Tst_mse1, tst_mse1)
        
        
        tst_mse2 = evaluation(net, tst_iter2)
        tst_mse2 = tst_mse2.cpu().numpy()
        Tst_mse2 = np.append(Tst_mse2, tst_mse2)
        
        
        print(f'train mse {trn_mse:.3f}, '
              f'test1 mse {tst_mse1:.3f},'
              f'test2 mse {tst_mse2:.3f},')

    return  Trn_mse ,Tst_mse1, Tst_mse2






"""
# =============================================================================
# #========================01 Set Experiment(Time & Location)==================
# =============================================================================
"""

depth = 25
Batch_size = 512
dataset_trn = 'ECCO2'
dataset_tst1 = 'GLORYS'
dataset_tst2 = 'HYCOM'



LON = np.arange(0.125, 360.125, 1)  
LAT = np.arange(-57.375, 58.625, 1)


num_lon = len(LON)
num_lat = len(LAT)



"""
# =============================================================================
# #=======================02 Load Data(training & test datasets)===============
# =============================================================================
"""

# =================================training dataset============================
# Sea surface height anomaly
filename='./'+dataset_trn+'/SSHA/'+'ssha.mat'
data=h5py.File(filename,mode='r')
SSHA = data['ssha'][:]

# Sea surface temperature anomaly
filename='./'+dataset_trn+'/SSTA/'+'ssta.mat'
data=h5py.File(filename,mode='r')
SSTA = data['ssta'][:]
    
# Subsurface temperature anomaly
filename='./'+dataset_trn+'/'+str(depth)+'m/'+'SUBTA.mat'
data=h5py.File(filename,mode='r')
SUBTA = data['subta'][:]


# Normalization
SSHA = standardization(SSHA,SSHA)
SSTA = standardization(SSTA,SSTA)

XX_TRN = np.stack((SSTA,SSHA),axis=1) 
XX_TRN = np.nan_to_num(XX_TRN)
SUBTA = np.nan_to_num(SUBTA)

XX_TRN = torch.tensor(XX_TRN, dtype=torch.float32)
YY_TRN = torch.tensor(SUBTA, dtype=torch.float32)

# API
train_ids = TensorDataset(XX_TRN, YY_TRN) 
train_iter_False = DataLoader(dataset=train_ids, batch_size=Batch_size, shuffle=False)  
trn_iter_shuffle = DataLoader(dataset=train_ids, batch_size=Batch_size, shuffle=True)





# ================================= tst1 dataset ==============================
# Sea surface height anomaly
filename='./'+dataset_tst1+'/SSHA/'+'ssha.mat'
data=h5py.File(filename,mode='r')
SSHA1 = data['ssha'][:]

# Sea surface temperature anomaly
filename='./'+dataset_tst1+'/SSTA/'+'ssta.mat'
data=h5py.File(filename,mode='r')
SSTA1 = data['ssta'][:]

# Subsurface temperature anomaly
filename='./'+dataset_tst1+'/'+str(depth)+'m/'+'SUBTA.mat'
data=h5py.File(filename,mode='r')
SUBTA1 = data['subta'][:]


SSHA1 = standardization(SSHA1,SSHA1)
SSTA1 = standardization(SSTA1,SSTA1)


XX_tst1 = np.stack((SSTA1,SSHA1),axis=1)    
YY_tst1 = SUBTA1 

XX_tst1 = np.nan_to_num(XX_tst1)
YY_tst1 = np.nan_to_num(YY_tst1)

XX_tst1 = torch.tensor(XX_tst1, dtype=torch.float32)
YY_tst1 = torch.tensor(YY_tst1, dtype=torch.float32)

tst_ids1 = TensorDataset(XX_tst1, YY_tst1) 
tst_iter1 = DataLoader(dataset=tst_ids1, batch_size=Batch_size, shuffle=False)





# ================================= tst2 dataset ==============================
# Sea surface height anomaly
filename='./'+dataset_tst2+'/SSHA/'+'ssha.mat'
data=h5py.File(filename,mode='r')
SSHA2 = data['ssha'][:]

# Sea surface temperature anomaly
filename='./'+dataset_tst2+'/SSTA/'+'ssta.mat'
data=h5py.File(filename,mode='r')
SSTA2 = data['ssta'][:]

# Subsurface temperature anomaly
filename='./'+dataset_tst2+'/'+str(depth)+'m/'+'SUBTA.mat'
data=h5py.File(filename,mode='r')
SUBTA2 = data['subta'][:]


SSHA2 = standardization(SSHA2,SSHA2)
SSTA2 = standardization(SSTA2,SSTA2)

XX_tst2 = np.stack((SSTA2,SSHA2),axis=1)     
YY_tst2 = SUBTA2

XX_tst2 = np.nan_to_num(XX_tst2)
YY_tst2 = np.nan_to_num(YY_tst2)

XX_tst2 = torch.tensor(XX_tst2, dtype=torch.float32)
YY_tst2 = torch.tensor(YY_tst2, dtype=torch.float32)

tst_ids2 = TensorDataset(XX_tst2, YY_tst2) 
tst_iter2 = DataLoader(dataset=tst_ids2, batch_size=Batch_size, shuffle=False)



'''
# =============================================================================
# #===========================03 Construction framework========================
# =============================================================================
'''

channel = 2

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, channel, num_lat, num_lon)
    
    
net = torch.nn.Sequential(
    Reshape(), 
    nn.Conv2d(2, 16, kernel_size=3,padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3,padding=1), nn.ReLU(),
    nn.Conv2d(32, 1, kernel_size=3,padding=1))


X = torch.rand(size=(1, channel, num_lat, num_lon), dtype=torch.float32)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)



  
num_epochs = 100


Trn_mse, Tst_mse1, Tst_mse2 = train_CNN_model(net, trn_iter_shuffle,tst_iter1, tst_iter2,
                                           num_epochs, num_lat,num_lon, dataset_trn, dataset_tst1, dataset_tst2)


 # for the 1st test dataset 
savepath = './'+dataset_trn+'-'+dataset_tst1+'/'
filename = savepath + 'Y_'+str(depth)+'.mat'
data_generation(net, tst_iter1, num_lat, num_lon, filename)  

# for the 2nd test dataset 
savepath = './'+dataset_trn+'-'+dataset_tst2+'/'
filename = savepath +  'Y_'+str(depth)+'.mat'
data_generation(net, tst_iter2, num_lat, num_lon, filename)  
    
print('--------------------------------------------------------')

savepath1 = './'+dataset_trn+'-'+dataset_trn+'/'
scio.savemat(savepath1+'Trn_mse'+str(depth)+'.mat',{'Trn_mse':Trn_mse})  
savepath2 = './'+dataset_trn+'-'+dataset_tst1+'/'
scio.savemat(savepath2+'Tst_mse1'+str(depth)+'.mat',{'Tst_mse1':Tst_mse1})  
savepath3 = './'+dataset_trn+'-'+dataset_tst2+'/'
scio.savemat(savepath3+'Tst_mse2'+str(depth)+'.mat',{'Tst_mse2':Tst_mse2})


torch.save(net.state_dict(), 'sequential_model_'+str(depth)+'.pth')

