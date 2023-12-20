
import numpy as np
import math
import h5py  
from mpi4py import MPI
from datetime import date
import scipy.io as scio  
import GSVC_fun as GSVC






def Calculate_beta_by_GSVC(rank,LON,LAT,width_time,width_space,depth):

    '''
    
    Parameters
    ----------
    rank : The unique identifier assigned to each process in the MPI communicator.
    LON : The longitude of the area to be calculated
    LAT : The latitude of the area to be calculated
    width_time : Time bandwidth set by the GSVC model
    width_space : Space bandwidth set by the GSVC model, Meridional bandwidth
    depth : depth

    Returns
    -------
    BETA : Coefficient of the model

    '''    
    
    
    # ------------------- 1 Experimental parameter setting  -------------------
    
    # Select a longitude according to rank and calculate the coefficients.
    # You can also choose multiple columns to calculate, depending on the task allocation of each thread in your parallel
    lon=LON[rank]
    length=len(lon)
    dim = 3

    # time :
    year_trn_begin  =  1994
    year_trn_end    =  2019

    # Longitude and latitude range of the dataset: data accuracy is 0.5° * 0.5°
    acc = 0.5
    XX  =  np.arange(210.125, 220.125, acc)
    YY  =  np.arange(0.125, 5.125, acc)
     
    # Calculate the amount of data based on the bandwidth
    width_time   =  2 * width_time  
    width_space  =  int( 1/acc * width_space * 2 ) # The amount of data corresponding to the Meridional bandwidth
    
    
    
    
    
    
    
    
    
    # ---------- 2 Compute weight matrix : Gaussian kernel function  ----------
   
    index_time = 2 * width_time + 1
    index_lon = 4 * width_space + 1
    index_lat = 2 * width_space + 1
       
    dataw = np.full((index_lon, index_lat, index_time), np.nan)
    wlon = np.linspace(-2, 2, index_lon)
    wlat = np.linspace(-2, 2, index_lat)
    wtime = np.linspace(-2, 2, index_time)
       
    # Gaussian weighting function
    exp_wlon = np.exp(-wlon**2)
    exp_wlat = np.exp(-wlat**2)
    
    for t in range(index_time):
        dataw[:, :, t] = np.exp(-wtime[t]**2) * np.outer(exp_wlon, exp_wlat)
    
    dataw[dataw < 0.01] = np.nan
    # Check if np.nan is of type float
    assert isinstance(np.nan, float)
    
    
    
    
    
    
    
    
    
    # ---------------- 3 Find the corresponding calendar day ------------------
    
    # Find the date corresponding to the same calendar day from 1 to 365
    calendar_day365 = np.array([date.toordinal(date(Year, 1, 1)) - date.toordinal(date(year_trn_begin, 12, 31)) for Year in range(year_trn_begin + 1, year_trn_end + 1)])
    calendar_day365 = calendar_day365-1
    
    
    # Find the date corresponding to the same calendar day for 366   
    calendar_day366 = np.array([date.toordinal(date(Year, 12, 31)) - date.toordinal(date(year_trn_begin - 1, 12, 31)) for Year in range(year_trn_begin, year_trn_end + 1) if (np.mod(Year, 4) == 0)])
    
    
    
    
    
    
    
    
    
    # ------------------ 4 Load data (SSTA / SSHA /SUBTA)  --------------------
     
    # Judge the block position : the size of each piece of raw data is 340 * 120, totally 6 pieces.
    N    =  np.where(XX==LON[rank])[0]
    part =  math.floor(N/120)+1
    
    width_data_block  =  120
    height_data_block  =  340
    time   =  np.arange(date.toordinal(date(1993, 1, 1)), date.toordinal(date(2021, 1, 1)),1)-date.toordinal(date(1993, 1, 1))
    
    
    # Select data area and index on each part(xx1 xx2)
    n1  =  np.where(XX==LON[rank])[0]
    N1  =  np.mod(n1,120)
    xx1 =  N1-2*width_space
    xx2 =  N1+2*width_space
    xcoord  =  np.arange(xx1,xx2+1,1)


    # Determine whether the numbers need to be spliced
    if xcoord[0]<0:
        # Front piece
        J  = [part-1, part]
        p1 = xcoord[np.where(xcoord<0)[0]] + width_data_block
        p2 = xcoord[np.where(xcoord>-1)[0]]

    elif xcoord[-1]>119:
        #Back piece
        J  = [part, part+1]
        p1 = xcoord[np.where(xcoord<120)[0]]
        p2 = xcoord[np.where(xcoord>119)[0]] - width_data_block  

    else:
        #unspliced
        J  = [part]
        p1 = xcoord[np.where(xcoord<119)[0]]
        p2 = 0
        
    J = np.array(J)
    J[J < 1] += 6
    J[J > 6] -= 6

    p=[p1,p2]
    

    

    height =  width_space*2+height_data_block
    width  =  width_space*4+1 
    
    SSHA    =   np.full([height, width,len(time)], np.nan)
    SSTA    =   np.full([height, width,len(time)], np.nan)
    SUBTA   =   np.full([height, width,len(time)], np.nan)
    
    
    add=0
    for i in range(len(J)): 
        
        # Sea surface height anomaly
        filename='./SSHA/'+'ssha'+str(J[i])+'.mat'
        data=h5py.File(filename,mode='r')
        data = data['ssha'][:]
        data= data.transpose(2,1,0)
        SSHA[width_space:width_space+height_data_block,add:add+len(p[i]),:]=data[:,p[i],:]
    
        # Sea surface temperature anomaly
        filename='./SSTA/'+'ssta'+str(J[i])+'.mat'
        data=h5py.File(filename,mode='r')
        data = data['ssta'][:]
        data= data.transpose(2,1,0)
        SSTA[width_space:width_space+height_data_block,add:add+len(p[i]),:]=data[:,p[i],:]
        
        # Subsurface temperature anomaly
        filename='./SUBTA/'+str(depth)+'m/'+'subta'+str(J[i])+'.mat'
        data=h5py.File(filename,mode='r')
        data = data['subta'][:]
        data= data.transpose(2,1,0)
        SUBTA[width_space:width_space+height_data_block,add:add+len(p[i]),:]=data[:,p[i],:]
        
        add=len(p[i])
      
        
      
        
      
        
      
        
      
    # ----------- 5 Calculate the coefficients of the GSVC model  -------------
    
    # Used to store results
    BETA  =   np.full([len(LAT),length,366,dim], np.nan)

    
    # Start with data0
    date0=date.toordinal(date(year_trn_begin,1,1))-date.toordinal(date(1993,1,1)) 
    
    # =============================================================================           
    #  for calendar_day3665
    # =============================================================================
 
    dataw_rep = np.tile(dataw, (1, 1, len(calendar_day365)))
    # Weight matrix for repeated samples
    
    for t1 in range(365):  
    
        for j in range(len(LAT)):      
    
            lat=LAT[j]
      
            n = np.where(YY==lat)[0]
            x_c   =   math.floor(width/2)
            y_c = n+width_space
    
            # Determine if the spot is land
            e1 = np.isnan(SSTA[y_c, x_c, 1]).astype(int)
            e2 = np.isnan(SSHA[y_c, x_c, 1]).astype(int)
            e3 = np.isnan(SUBTA[y_c, x_c, 1]).astype(int)

            exist=e1*e2*e3
    
            
            if exist==0:
                continue
            else:
                
                #Select data on same calendar day of each year within the bandwidth
                
                #+++++++++++++++++++++++++++  time  ++++++++++++++++++++++++++
                TIME = np.concatenate([np.arange(date0 + t1 + calendar_day365[k] - width_time, date0 + t1 + calendar_day365[k] + width_time + 1, 1) for k in range(len(calendar_day365))])

                #+++++++++++++++++++++++++++Set location++++++++++++++++++++++++++ 
                ycoord = np.arange(y_c-1*width_space,y_c+1*width_space+1,1)
                
                # ++++++++++++++++++++++++++Process X and Y++++++++++++++++++++++++++
                X_SSTA    =   SSTA[ycoord,:,TIME]
                X_SSHA    =   SSHA[ycoord,:,TIME]
                Y_SUBTA   =   SUBTA[ycoord,:,TIME]
               
                
                BETA[j, 0, t1, :] = GSVC.GSVC(X_SSTA, X_SSHA, Y_SUBTA, dataw_rep).flatten()

                
                
                
                
    # =============================================================================           
    #  for calendar_day366  
    # =============================================================================
    dataw_rep = np.tile(dataw, (1, 1, len(calendar_day366)))
    # Weight matrix for repeated samples  

    for j in range(len(LAT)):
        
        lat=LAT[j]
        
        n = np.where(YY==lat)[0]
        x_c   =   math.floor(width/2)
        y_c = n+width_space
    
        # Determine if the spot is land
        e1 = np.isnan(SSTA[y_c, x_c, 1]).astype(int)
        e2 = np.isnan(SSHA[y_c, x_c, 1]).astype(int)
        e3 = np.isnan(SUBTA[y_c, x_c, 1]).astype(int)
    
        exist=e1*e2*e3
        
        if exist==0:
            continue
        else:
            
            #Select data on same calendar day of each year within the bandwidth
            
            #+++++++++++++++++++++++++++  time  ++++++++++++++++++++++++++
            TIME = np.concatenate([np.arange(date0 + t1 + calendar_day366[k] - width_time, date0 + t1 + calendar_day366[k] + width_time + 1, 1) for k in range(len(calendar_day366))])

            #+++++++++++++++++++++++++++Set location++++++++++++++++++++++++++ 
            ycoord = np.arange(y_c-1*width_space,y_c+1*width_space+1,1)
            
            # ++++++++++++++++++++++++++Process X and Y++++++++++++++++++++++++++
            X_SSTA    =   SSTA[ycoord,:,TIME]
            X_SSHA    =   SSHA[ycoord,:,TIME]
            Y_SUBTA   =   SUBTA[ycoord,:,TIME]
           

            BETA[j, 0, 365, :] = GSVC.GSVC(X_SSTA, X_SSHA, Y_SUBTA, dataw_rep).flatten()
          
    
    return BETA




'''
# =============================================================================
# ================================  MPI parallel===============================
# =============================================================================                 
'''  

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


depth=50
dataset='GLORYS'
print(depth)

#Select an area : 
LON  = np.arange(0.125, 360.125, 1)
LAT  = np.arange(-57.375, 58.625, 1)

# Set bandwidth:
width_time   =  30
width_space  =  1 # Meridional bandwidth, the zonal bandwidth is twice the meridional bandwidth

#Assign tasks to each thread
n   = math.ceil(len(LON)/size) 

#For the each thread
print(rank)
num = range(rank*n,(rank+1)*n)  

for i in range(n):

    result=Calculate_beta_by_GSVC(num[i],LON,LAT,width_time,width_space,depth)
    filename='./'+dataset+'/'+str(depth)+'m/'+str(num[i])+'_beta.mat'
    scio.savemat(filename,{'beta':result})  






