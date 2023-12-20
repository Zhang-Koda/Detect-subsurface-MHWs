import numpy as np
import scipy.io as scio 
import h5py 
from datetime import date
import math
import statsmodels.api as sm





for p in range(1,7):
  
    depth=50
    dataname='GLORYS'
    
    SIZE=60;
    LON  = np.arange((p-1)*SIZE+0.125, p*SIZE+0.125, 1)
    LAT  = np.arange(-59.375, 60.625, 1)

    # Latitude and longitude range :
    XX  =  np.arange(0.125, 359.875, 0.5)
    YY  =  np.arange(-79.875, 89.875, 0.5)
   
    # Regression dimension : 2(Number of variables) + 1(Constant term)
    dim = 3
         
    # Training set :
    year_trn_begin  =  1994
    year_trn_end    =  2019

    time_trn = np.arange(date.toordinal(date(year_trn_begin, 1, 1)), date.toordinal(date(year_trn_end, 12, 31)) + 1, dtype=int) - date.toordinal(date(1992, 1, 1))
  

    # ----------------1 Load data (SSTA\SSHA\SUBTA)  --------------------
     
    # Judge the block position : the size of each piece of raw data is 360 * 120
    N    =  np.where(XX==LON[0])[0]
    
    part =  math.floor(N/120)+1
    
        
    # Sea surface height anomaly
    filename='./'+dataname+'/SSHA/'+'ssha'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssha'][:]
    SSHA= data.transpose(2,1,0)
       
    # Sea surface temperature anomaly
    filename='./'+dataname+'/SSTA/'+'ssta'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['ssta'][:]
    SSTA= data.transpose(2,1,0)
    
    # Subsurface temperature anomaly
    filename='./'+dataname+'/'+str(depth)+'m/'+'SUBTA'+str(part)+'.mat'
    data=h5py.File(filename,mode='r')
    data = data['subta'][:]
    SUBTA= data.transpose(2,1,0)
       
    
    
    
    
    # ------------------  2 OLS model calculation  ----------------------------
    
    # Used to store results
    BETA  =   np.full([len(LAT),len(LON),1,dim], np.nan)

    N_lat =   len(LAT) 
    N_lon =   len(LON) 
       
    for i in range(N_lon):
        
        lon=LON[i]
        
        for j in range(N_lat):      
    
            lat=LAT[j]
            
            x_c = np.where(XX==lon)[0]
            y_c = np.where(YY==lat)[0]
            
            x_c =np.mod(x_c,120)
          
            # Determine if the spot is land
            e1 = np.isnan(SSTA[y_c, x_c, 1]).astype(int)
            e2 = np.isnan(SSHA[y_c, x_c, 1]).astype(int)
            e3 = np.isnan(SUBTA[y_c, x_c, 1]).astype(int)
            exist=e1*e2*e3
            
            if exist==0:
             
                continue
            else:
                 
                # ++++++++++++++++++++++++++Process X and Y++++++++++++++++++++++++++
                X_SSTA    =   SSTA[y_c,x_c,time_trn]
                X_SSHA    =   SSHA[y_c,x_c,time_trn]
                Y_SUBTA =   SUBTA[y_c,x_c,time_trn]
                 
                X_SSTA    =  X_SSTA.reshape(-1,1)
                X_SSHA    =  X_SSHA.reshape(-1,1)
                Y_SUBTA =  Y_SUBTA.reshape(-1,1)
                
                e1=np.where(np.isnan(X_SSTA),0,1)
                e2=np.where(np.isnan(X_SSHA),0,1)
                e3=np.where(np.isnan(Y_SUBTA),0,1)
                
                index=e1*e2*e3
                index=np.where(index==1)[0]
                
    
                X_SSTA     =  X_SSTA[index]
                X_SSHA     =  X_SSHA[index]
                Y_SUBTA    =  Y_SUBTA[index]
    
                
                X      =  np.hstack((X_SSTA,X_SSHA,np.ones(X_SSTA.shape)))  
                Y      =  Y_SUBTA
                
                
                mod = sm.OLS(Y,X)
                result = mod.fit()

                BETA[j,i,0,:] = result.params
        
      
    filename='./'+dataname+'/'+str(depth)+'m/'+str(p)+'_beta.mat'  
    scio.savemat(filename,{'beta':BETA})  
       

    
