%% init

clear all; close all;

gpml_path = '/nfs/arch11/researchData/USER/nhutting/code/gpml-matlab-master/';

restoredefaultpath;

disp('+Adding GPML toolbox to search path')
addpath(genpath(gpml_path))
run([gpml_path,'startup.m']);


base_path = get_data_dir(matlab.desktop.editor.getActiveFilename);
cd(base_path);
base_path = get_data_dir(matlab.desktop.editor.getActiveFilename);
export_path = [base_path,'exports/'];


%% EXPERIMENT 2: fitting a cos^4 based on temporal inputs - interpolation
%  NOTE: this usually works better; it is what people refer to as Gaussian Process Regression. Useful if you want to e.g. smooth out time series.

Nt = 1000;

% generate data
x = [1:Nt].';
noise_std = 0.01;
noise = randn(size(x,1),size(x,2))*0.01; 
y = cos(x./Nt.*2.*pi).^4 ;
x = [x,[gradient(y,1,1)]];
y = y+ noise + 5;
figure;plot(y);

% GP settings
pars=[];
pars.visualize = 1;
pars.shuffle_data = 1;
pars.Ntrain = 100;
pars.train_noise_std = 0;               % play around with this setting if you don't get statisfactory results
pars.GP.noise_std = noise_std;          % play around with this setting if you don't get statisfactory results
pars.demean_targets = 1;                % play around with this setting if you don't get statisfactory results
pars.normalize_inputs = 0;              % play around with this setting if you don't get statisfactory results
pars.GP.covfunc = {@covMaternard,3};    % kernel with automatic relevance determination, useful if you have a lot of inputs but only some are most likely actually meaningful for the predictions
pars.GP.hyp=struct('mean', [], 'cov', zeros(1,size(x,2)+1), 'lik',log(pars.GP.noise_std)); % note that for ARD it increases the number of COV parameters by the number of input dimension.

% GP training and visualization
TrainGaussianProcess(x,y,pars)
