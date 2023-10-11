%% init

clear all; close all;
gpml_path = '/nfs/arch11/researchData/USER/nhutting/code/gpml-matlab-master/';

restoredefaultpath;

disp('+Adding GPML toolbox to search path')
addpath(genpath(gpml_path))
run([gpml_path,'startup.m']);


base_path = get_data_dir(matlab.desktop.editor.getActiveFilename);
base_path = get_data_dir(matlab.desktop.editor.getActiveFilename);
cd(base_path);
export_path = [base_path,'exports/'];

%% EXPERIMENT 1: fitting a cos^4 purely based on temporal inputs - extrapolation
%  NOTE: this is quite hard because all we have as inputs is time.

% generate data
Nt = 1000;
x = [1:Nt].';
noise_std = 0.01;
noise = randn(size(x,1),size(x,2))*0.01; 
y = cos(x./Nt.*2.*pi).^4 ;
x = [x,[gradient(y,1,1)]];
y = y+noise;
figure;plot(y);

% GP settings
pars=[];
pars.Ntrain = 500;
pars.visualize = 1;
pars.train_noise_std = 0;
pars.GP.noise_std = noise_std;
pars.demean_targets = 1;
pars.normalize_inputs = 0;

% GP training
TrainGaussianProcess(x,y,pars)