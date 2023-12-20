This is the package for subsurface Marine Heatwave (MHW) detection, using sea surface temperature anomaly and sea surface height anomaly as predictors. The implemented methods include a geographically and seasonally varying coefficient (GSVC) linear regression, CNN classification (CNN_cla) learning, CNN regression (CNN_reg) learning, and ordinary least square (OLS) regression.

The package structure is outlined as follows:

1. **GSVC Model:** Estimate subsurface temperature anomaly (T') using the GSVC model (implemented in GSVC_main.py and GSVC_fun.py). The algorithm employs MPI parallelization for efficiency. Subsequently, pointwise MHWs are detected following the methodology of Hobday et al., 2016 (detect.m).

2. **CNN_cla Model:** Directly detect subsurface MHWs using the CNN_cla model, which generates a binary output (implemented in CNN_cla.py).

3. **CNN_reg Model:** Estimate T' using the CNN_reg model (implemented in CNN_cla.py) and then detect pointwise MHWs.

4. **OLS Model:** Train an OLS model at each grid to estimate T' and subsequently detect pointwise MHWs.

We will update to design an example for a specific small area. Welcome any criticism and corrections.
