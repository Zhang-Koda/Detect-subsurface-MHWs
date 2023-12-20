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
from datetime import date
from d2l import torch as d2l
from datetime import timedelta





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()  #Clear cuda video memory



'''
# =============================================================================
# #==================================== Funtion================================
# =============================================================================
'''



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
        
    comp_data_flat = comp_data.flatten()
    comp_data_flat = comp_data_flat[~np.isnan(comp_data_flat)]

    mean_value = np.mean(comp_data_flat)
    std_deviation = np.std(comp_data_flat)

    if std_deviation == 0:
        raise ValueError("Standard deviation is zero. Cannot perform standardization.")

    standardized_data = (input_data - mean_value) / std_deviation
    return standardized_data




def evaluate_accuracy_gpu(net, data_iter, device=None): 
    '''
    Evaluate the accuracy of a neural network model on a dataset using GPU acceleration.

    Parameters:
    - net (nn.Module): The neural network model.
    - data_iter: The data iterator.
    - device (torch.device, optional): The device on which to perform computations. Defaults to None.

    Returns:
    - float: The accuracy of the model on the dataset.
    
    '''
    
    if isinstance(net, nn.Module):
        net.eval() 
        if not device:
            device = next(iter(net.parameters())).device
            
    metric = d2l.Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            
            metric.add(d2l.accuracy(net(X), y), y.numel())   
            
    accuracy = metric[0] / metric[1]
    
    return accuracy



def obtain_confusion_matrix(net, data_iter, num_lat, num_lon):

    '''
    Obtain confusion matrix: A for TP, B for FN, C for FP, D for TN.

    Parameters:
    - net (nn.Module): The neural network model.
    - data_iter (DataLoader): The data iterator.
    - num_lat (int): The number of latitude values.
    - num_lon (int): The number of longitude values.
    - device (torch.device): The device on which to perform computations.

    Returns:
    - tuple: A tuple containing the elements of the confusion matrix (A, B, C, D).
    '''

    if isinstance(net, nn.Module):
        net.eval() 

    A = torch.zeros(num_lat, num_lon, device=device)
    B = torch.zeros(num_lat, num_lon, device=device)
    C = torch.zeros(num_lat, num_lon, device=device)
    D = torch.zeros(num_lat, num_lon, device=device)

    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):
            X = X.to(device)
            Y_true = y.to(device)
            y_hat = net(X)

            probabilities = torch.sigmoid(y_hat)
            Y_pred = (probabilities[:, 0, :, :] <= probabilities[:, 1, :, :]).int()

            temp = Y_true - Y_pred

            A += torch.sum((temp == 0) & (Y_true == 1), dim=0)
            B += torch.sum((temp == -1), dim=0)
            C += torch.sum((temp == 1), dim=0)
            D += torch.sum((temp == 0) & (Y_true == 0), dim=0)

    return A, B, C, D   

    
 


def evaluate_loss_gpu(net, data_iter,loss, device=None): 
    '''
    Evaluate the average loss of a neural network on a given dataset.

    Parameters:
    - net : The neural network model.
    - data_iter : The data iterator.
    - loss : The loss funtion.
    - device : The device on which to perform computations.

    Returns:
    - float: The average loss.
    '''
    num_batches = len(data_iter)
      
    if isinstance(net, nn.Module):
        net.eval() 
        if not device:
            device = next(iter(net.parameters())).device
 
    metric = d2l.Accumulatorz(2)
      
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            
            metric.add(loss(net(X), y), y.numel())   
            
    return metric[0] / num_batches



def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)  #variance to remain the same

def train_CNN_model(net, train_iter, tst1_iter, tst2_iter, num_epochs, num_lat, num_lon):    
    
    
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
    - device : The device on which to perform computations.

    Returns:
    - net : The trained CNN model.
    - trn_loss : The mse on training data iterator.
    - tst1_loss : The mse on the first test data iterator.
    - tst2_loss : The mse on the second test data iterator.
    
    '''
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    
    area = num_lat * num_lon
    
    # Give different weights to different classes in the loss function to 
    # make the model pay more attention to the 'one' class.
    weights = torch.tensor([0.20, 0.80], device=device) 
    loss = nn.CrossEntropyLoss(weight=weights)  
    optimizer = torch.optim.Adadelta(net.parameters())


    timer = d2l.Timer()
    
    
    tst1_loss = np.empty(0)
    tst2_loss = np.empty(0)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
       
        metric = d2l.Accumulator(3)
        net.train()


        for i, (X, y) in enumerate(train_iter):  
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)


            y_hat = net(X)
            
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()   
      
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y)/area, X.shape[0])
            timer.stop()
    
            trn_l = metric[0] / metric[2]
            trn_acc = metric[1] / metric[2]   
        
        
        tst1_acc = evaluate_accuracy_gpu(net, tst1_iter)
        tst2_acc = evaluate_accuracy_gpu(net, tst2_iter)

        tst1_loss = np.append(tst1_loss, evaluate_loss_gpu(net, tst1_iter,loss))
        tst2_loss = np.append(tst2_loss, evaluate_loss_gpu(net, tst2_iter,loss))
            
       

    print(f'trn_loss {trn_l:.3f},trn_acc {trn_acc:.3f},tst1_acc {tst1_acc:.3f}, tst2_acc {tst2_acc:.3f}')
 
        
    return net, trn_l, tst1_loss, tst2_loss



"""
# =============================================================================
# #========================01 Set Experiment(Time & Location)==================
# =============================================================================
"""


depth = 25
Batch_size = 365
dataset_trn = 'GLORYS'
dataset_tst1 = 'ECCO2'
dataset_tst2 = 'HYCOM'



LON = np.arange(0.125, 360.125, 1)
LAT = np.arange(-59.375, 60.625, 1)

num_lon = len(LON)
num_lat = len(LAT)


# X & Y coordinates for training and test datasets
XX_1  =  np.arange(0.125, 359.875, 0.5)
YY_1  =  np.arange(-79.875, 89.875, 0.5)
XX_2  =  np.arange(0.125, 359.875, 0.5)
YY_2  =  np.arange(-89.875, 89.875, 0.5)
XX_3  =  np.arange(0.125, 359.875, 0.5)
YY_3  =  np.arange(-59.875, 59.875, 0.5)



# time for training and test datasets
year_from1 = 1993
year_trn_begin  =  1994
year_trn_end    =  2019

year_from2 = 1992
year_tst1_begin  =  1994
year_tst1_end    =  2019

year_from3 = 1993
year_tst2_begin  =  1994
year_tst2_end    =  2011

time_trn = np.arange(date.toordinal(date(year_trn_begin, 1, 1)), date.toordinal(date(year_trn_end, 12, 31) + timedelta(days=1))) - date.toordinal(date(year_from1, 1, 1))
time_tst1 = np.arange(date.toordinal(date(year_tst1_begin, 1, 1)), date.toordinal(date(year_tst1_end, 12, 31) + timedelta(days=1))) - date.toordinal(date(year_from2, 1, 1))
time_tst2 = np.arange(date.toordinal(date(year_tst2_begin, 1, 1)), date.toordinal(date(year_tst2_end, 12, 31) + timedelta(days=1))) - date.toordinal(date(year_from3, 1, 1))

time_trn = time_trn.astype(int)
time_tst1 = time_tst1.astype(int)
time_tst2 = time_tst2.astype(int)




"""
# =============================================================================
# #=======================02 Load Data(training & test datasets)===============
# =============================================================================
"""

# =================================training dataset============================

SSHA   =   np.full([len(time_trn),num_lat, num_lon], np.nan)
SSTA    =   np.full([len(time_trn),num_lat, num_lon], np.nan)
SMHW     =   np.full([len(time_trn),num_lat, num_lon], np.nan)

M1    =  np.where(YY_1==LAT[0])[0]
M2    =  np.where(YY_1==LAT[-1])[0]
N1    =  np.where(XX_1==LON[0])[0]
N2    =  np.where(XX_1==LON[-1])[0]


print('load training dataset')
for i in range(6): 
    
    part = i + 1
    
    # Sea surface height anomoly
    filename='./'+dataset_trn+'/SSHA/'+'ssha'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssha'][:]
    data= data.transpose(0,2,1)
    SSHA[:,:,i*60:(i+1)*60]=data[time_trn,M1[0]:M2[0]+1:2,::2]
    
    # Sea surface temperature anomoly
    filename='./'+dataset_trn+'/SSTA/'+'ssta'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssta'][:]
    data= data.transpose(0,2,1)
    SSTA[:,:,i*60:(i+1)*60]=data[time_trn,M1[0]:M2[0]+1:2,::2]
    
    # Subsurface marine heatwaves
    filename='./'+dataset_trn+'/'+str(depth)+'m/'+'smhw'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['smhw'][:]
    data= data.transpose(0,2,1)
    SMHW[:,:,i*60:(i+1)*60]=data[time_trn,M1[0]:M2[0]+1:2,::2]


# Normalization
SSHA = standardization(SSHA,SSHA)
SSTA = standardization(SSTA,SSTA)

XX_TRN = np.stack((SSTA,SSHA),axis=1) 
XX_TRN = np.nan_to_num(XX_TRN)

XX_TRN = torch.tensor(XX_TRN, dtype=torch.float32)
YY_TRN = torch.tensor(SMHW, dtype=torch.long)

# API
train_ids = TensorDataset(XX_TRN, YY_TRN) 
trn_iter_non_shuffle = DataLoader(dataset=train_ids, batch_size=Batch_size, shuffle=False)  
trn_iter_shuffle = DataLoader(dataset=train_ids, batch_size=Batch_size, shuffle=True)






# ================================= tst1 dataset ==============================

SSHA1    =   np.full([len(time_tst1),num_lat, num_lon], np.nan)
SSTA1    =   np.full([len(time_tst1),num_lat, num_lon], np.nan)
SMHW1    =   np.full([len(time_tst1),num_lat, num_lon], np.nan)

M1    =  np.where(YY_2==LAT[0])[0]
M2    =  np.where(YY_2==LAT[-1])[0]
N1    =  np.where(XX_2==LON[0])[0]
N2    =  np.where(XX_2==LON[-1])[0]


print('load tst1 dataset')
for i in range(6): 
    
    part = i + 1
    
    # Sea surface height anomoly
    filename='./'+dataset_tst1+'/SSHA/'+'ssha'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssha'][:]
    data= data.transpose(0,2,1)
    SSHA1[:,:,i*60:(i+1)*60]=data[time_tst1,M1[0]:M2[0]+1:2,::2]

    # Sea surface temperature anomoly
    filename='./'+dataset_tst1+'/SSTA/'+'ssta'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssta'][:]
    data= data.transpose(0,2,1)
    SSTA1[:,:,i*60:(i+1)*60]=data[time_tst1,M1[0]:M2[0]+1:2,::2]

    # Subsurface marine heatwaves
    filename='./'+dataset_tst1+'/'+str(depth)+'m/'+'smhw'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['smhw'][:]
    data= data.transpose(0,2,1)
    SMHW1[:,:,i*60:(i+1)*60]=data[time_tst1,M1[0]:M2[0]+1:2,::2]


SSHA1 = standardization(SSHA1,SSHA1)
SSTA1 = standardization(SSTA1,SSTA1)

XX_tst1 = np.stack((SSTA1,SSHA1),axis=1)     
YY_tst1 = SMHW1

XX_tst1 = np.nan_to_num(XX_tst1)
YY_tst1 = np.nan_to_num(YY_tst1)

XX_tst1 = torch.tensor(XX_tst1, dtype=torch.float32)
YY_tst1 = torch.tensor(YY_tst1, dtype=torch.long)

tst_ids1 = TensorDataset(XX_tst1, YY_tst1) 
tst1_iter = DataLoader(dataset=tst_ids1, batch_size=Batch_size, shuffle=False)



# ================================= tst2 dataset ==============================

SSHA2    =   np.full([len(time_tst2),num_lat, num_lon], np.nan)
SSTA2    =   np.full([len(time_tst2),num_lat, num_lon], np.nan)
SMHW2    =   np.full([len(time_tst2),num_lat, num_lon], np.nan)

M1    =  np.where(YY_3==LAT[0])[0]
M2    =  np.where(YY_3==LAT[-1])[0]
N1    =  np.where(XX_3==LON[0])[0]
N2    =  np.where(XX_3==LON[-1])[0]


print('load tst2 dataset')
for i in range(6): 
    
    part = i + 1
    
    # Sea surface height anomoly
    filename='./'+dataset_tst2+'/SSHA/'+'ssha'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssha'][:]
    data= data.transpose(0,2,1)
    SSHA2[:,:,i*60:(i+1)*60]=data[time_tst2,M1[0]:M2[0]+1:2,::2]

    # Sea surface temperature anomoly
    filename='./'+dataset_tst2+'/SSTA/'+'ssta'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssta'][:]
    data= data.transpose(0,2,1)
    SSTA2[:,:,i*60:(i+1)*60]=data[time_tst2,M1[0]:M2[0]+1:2,::2]

    # Subsurface marine heatwaves
    filename='./'+dataset_tst2+'/'+str(depth)+'m/'+'smhw'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['smhw'][:]
    data= data.transpose(0,2,1)
    SMHW2[:,:,i*60:(i+1)*60]=data[time_tst2,M1[0]:M2[0]+1:2,::2]


SSHA2 = standardization(SSHA2,SSHA2)
SSTA2 = standardization(SSTA2,SSTA2)


XX_tst2 = np.stack((SSTA2,SSHA2),axis=1)     
YY_tst2 = SMHW2

XX_tst2 = np.nan_to_num(XX_tst2)
YY_tst2 = np.nan_to_num(YY_tst2)

XX_tst2 = torch.tensor(XX_tst2, dtype=torch.float32)
YY_tst2 = torch.tensor(YY_tst2, dtype=torch.long)

tst_ids2 = TensorDataset(XX_tst2, YY_tst2) 
tst2_iter = DataLoader(dataset=tst_ids2, batch_size=Batch_size, shuffle=False)



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
    nn.Conv2d(2, 16, kernel_size=5,padding=2), nn.Sigmoid(),
    nn.Conv2d(16, 32, kernel_size=5,padding=2), nn.Sigmoid(),
    nn.Conv2d(32, 2, kernel_size=5,padding=2))


X = torch.rand(size=(1, channel, num_lat, num_lon), dtype=torch.float32)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
    







"""
# =============================================================================
# #===========================04 Training model & save ========================
# =============================================================================
"""

num_epochs =  100

# Training model
net, trn_loss, tst1_loss, tst2_loss = train_CNN_model(net, trn_iter_shuffle,tst1_iter,tst2_iter,num_epochs, num_lat,num_lon)



# Save loss
savepath1 = './'+dataset_trn+'-'+dataset_trn+'/'+str(depth)+'m/'
savepath2 = './'+dataset_trn+'-'+dataset_tst1+'/'+str(depth)+'m/'
savepath3 = './'+dataset_trn+'-'+dataset_tst2+'/'+str(depth)+'m/'
scio.savemat(savepath1+'trn_loss.mat',{'trn_loss':trn_loss})
scio.savemat(savepath2+'tst1_loss.mat',{'tst1_loss':tst1_loss})  
scio.savemat(savepath3+'tst2_loss.mat',{'tst2_loss':tst2_loss})




# Save confusion matrix for the first test dataset
A, B, C, D = obtain_confusion_matrix(net, tst1_iter ,num_lat,num_lon)

A = A.cpu().numpy()
B = B.cpu().numpy()
C = C.cpu().numpy()
D = D.cpu().numpy()

savepath = '.'+dataset_trn+'-'+dataset_tst1+'/'+str(depth)+'m/'

scio.savemat(savepath+'A.mat',{'A':A})  
scio.savemat(savepath+'B.mat',{'B':B})  
scio.savemat(savepath+'C.mat',{'C':C})  
scio.savemat(savepath+'D.mat',{'D':D})  




# Save confusion matrix for the second test dataset
A, B, C, D = obtain_confusion_matrix(net, tst2_iter, num_lat, num_lon)

A = A.cpu().numpy()
B = B.cpu().numpy()
C = C.cpu().numpy()
D = D.cpu().numpy()

savepath = './'+dataset_trn+'-'+dataset_tst2+'/'+str(depth)+'m/'

scio.savemat(savepath+'A.mat',{'A':A})  
scio.savemat(savepath+'B.mat',{'B':B})  
scio.savemat(savepath+'C.mat',{'C':C})  
scio.savemat(savepath+'D.mat',{'D':D})    

