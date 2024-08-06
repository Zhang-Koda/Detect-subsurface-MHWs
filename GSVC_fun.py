
import numpy as np
import statsmodels.api as sm



def GSVC(X1,X2,Y,weight):
    
    '''
    

    Enter the explanatory variables X1,X2 and the response variables Y, as well as the weight matrix weight. 
    Notice that their shape should be compatible.

    Return the coefficient of the GSVC model.

    '''
    assert X1.shape == X2.shape == Y.shape == weight.shape, "Input shapes are not compatible."

    # Data processing : straighten the data
    X1       =  X1.reshape(-1,1)
    X2       =  X2.reshape(-1,1)
    Y        =  Y.reshape(-1,1)
    weight   =  weight.reshape(-1,1)


    # Create a mask for NaN values
    nan_mask = ~np.isnan(X1) & ~np.isnan(X2) & ~np.isnan(Y) & ~np.isnan(weight)


    # Apply the mask to all variables
    X1       =  X1[nan_mask]
    X2       =  X2[nan_mask]
    Y        =  Y[nan_mask]
    weight   =  weight[nan_mask]
   
   
    # Define X & Y : assign weight
    X     =  np.column_stack((X1*np.sqrt(weight), X2*np.sqrt(weight), np.sqrt(weight)))
    Y     =  Y * np.sqrt(weight)
 
    
    # Create model
    mod = sm.OLS(Y,X)
    result = mod.fit()
    
    return result.params

