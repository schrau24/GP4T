## Gaussian Processes For Tracking

This repo contains minimal working examples on how to use Gaussian Processes for tracking in medical imaging. 

Relevant references include:
* Huttinga, Niek RF, et al. "Gaussian Processes for real-time 3D motion and uncertainty estimation during MR-guided radiotherapy." arXiv preprint arXiv:2204.09873 (2022)
* Huttinga, Niek RF, et al. "Real-time myocardial landmark-tracking for MRI-guided cardiac radio-ablation using Gaussian Processes." ISMRM 2022. https://index.mirasmart.com/ISMRM2022/PDFfiles/0014.html 
* Huttinga, Niek RF, et al. "Real-time quality assurance for volumetric motion estimation during MR-guided radiotherapy." ISMRM 2022. https://index.mirasmart.com/ISMRM2022/PDFfiles/2127.html 
* Rasmussen, Carl E, and Williams, Christopher KI. "Gaussian processes for machine learning." Vol. 1. Cambridge, MA: MIT press, 2006.
* Rasmussen, Carl E, and Nickisch, Hannes. "Gaussian processes for machine learning (GPML) toolbox." The Journal of Machine Learning Research 11 (2010): 3011-3015.

## Getting Started

### Installations
#### With Matlab

* Download the GPML toolbox by Rasmussen and Nickisch http://gaussianprocess.org/gpml/code/matlab/
* Make sure to run the `startup.m` script to include all required files in Matlab's search path

#### With Python

* Install the GPy toolbox https://github.com/SheffieldML/GPy

### Run the experiments

* `mwe_interp.m`: Simple Gaussian Process Regression that interpolates function values 
* `mwe_extrap.m`: Simple Gaussian Process Regression that extrapolates function values
* `mwe_cardiac_tracking.m`: Code that shows how to train a Gaussian Process to predict the location of a landmark based on just two readouts of MR data.
* `mwe_3d_dvf_tracking.m`: Example code that tracks 3D DVF coefficients. This requires the following dataset: https://surfdrive.surf.nl/files/index.php/s/iLkogKsPXx8pgdc
