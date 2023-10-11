%% init

clear all; close all;

gpml_path = '/nfs/arch11/researchData/USER/nhutting/code/gpml-matlab-master/';

restoredefaultpath;

disp('+Adding GPML toolbox to search path')
addpath(genpath(gpml_path))
run([gpml_path,'startup.m']);


base_path = get_data_dir(matlab.desktop.editor.getActiveFilename);
cd(base_path);
export_path = [base_path,'exports/'];


%% Load data
% FIRST DOWNLOAD THE DATASET FROM HERE: https://surfdrive.surf.nl/files/index.php/s/iLkogKsPXx8pgdc and adjust the path appropriately
data_path = [base_path,'/data/synth_timeresolved_3d/data.mat'];
load(data_path)


%% Visualize data

% visualize images [drag with left mouse button down]
slicer5d(images);
 
% visualize motion-fields [drag with left mouse button down]
slicer5d(dvfs)

%% Construct GP training set: 1) Computate projections of the images to construct the GP inputs

% make projections
projections_FH = squeeze(sum(sum(images,2),3));
projections_LR = squeeze(sum(sum(images,1),3));
projections_AP = squeeze(sum(sum(images,2),1));

projections = [projections_LR.',projections_FH.',projections_AP.'];


%% Construct GP training set: 2) Compress DVFs to obtain GP targets

N = size(dvfs,1);
[u,s,v]=svd(reshape(dvfs,N^3*3,[]),'econ');

% select the two largest principal components of the temporal dvf variation
targets=v(:,1:2);

figure;plot(targets);
%% Set training data parameters:

close all;

% GP parameters
pars=[];
pars.Ntrain=round(size(inputs,1)*0.5);
pars.train_indices = 1:pars.Ntrain;
compress_input_data = 0; % <----- 0: use all samples on the spokes, 1: compress the spokes to reduce input dimensionality


%% Optional: prior to fitting the GP, compress the input data

if compress_input_data
    % compute an input data compression with PCA:
    train_projections = projections(pars.train_indices,:); % select only TRAIN (!) projections
    train_mean = mean(projections(pars.train_indices,:),1); % demean only TRAIN (!) projections

    % compute SVD of demeaned TRAIN (!) projections
    [u_x,s_x,v_x]=svd(train_projections.'-train_mean.','econ');

    % demean inputs and project on the projections basis for dimensionality reduction
    proj_basis = u_x*diag(1./diag(s_x));
    inputs = proj_basis'*(projections.'-train_mean.');

    % select more dimension as input than as output
    inputs = inputs(1:5,:).';
else
    inputs = projections;
end


%% visualize training data

figure; 
subplot(211);
if compress_input_data
    plot(inputs-mean(inputs));
    title('Input data');
else
    plot(inputs(:,60:70)-mean(inputs(:,60:70)));
    title('Some input data');
end

xlabel('Time indices');
ylabel('Magnitude [a.u.]');

subplot(212);
plot(targets);
title('Targets');
xlabel('Time indices');
ylabel('Magnitude [a.u.]');

%% Fit GP:

pars.train_noise_std = 0;
pars.visualize=1;
pars.demean_targets = 1;
pars.normalize_inputs = 1;
pars.GP.covfunc = {@covMaternard,3};
pars.GP.Niter = 3000;
noise_init = 2e-3; % initialization of the noise <--- note that it's quite sensitive to this!
pars.GP.hyp=struct('mean', [], 'cov', zeros(1,size(inputs,2)+1), 'lik',log(noise_init));


ntargets = size(targets,2);
for i=1:ntargets
    pars.GP.noise_std=max(max(abs(demean(targets(:,i))))*2*.04,1e-5);
    [hyp_opt{i},yhat(:,i),y_uncertainties(:,i),means(i)]=TrainGaussianProcess( inputs , targets(:,i) , pars);
end


%% evaluate the performance on the test data

% principal components
figure;
sgtitle('Evaluation of performance @ test time');
subplot(211);
plot(yhat(max(pars.train_indices)+1:end,1)+means(1));hold on;plot(targets(max(pars.train_indices)+1:end,1),'--*');legend('GP approximation','Target')
title('Principal component #1');

subplot(212);
plot(yhat(max(pars.train_indices)+1:end,2)+means(2));hold on;plot(targets(max(pars.train_indices)+1:end,2),'--*');legend('GP approximation','Target')
title('Principal component #2');

% motion-fields
dvf_approx = u(:,1:ntargets)*s(1:ntargets,1:ntargets)*targets.';
disp(['Relative error on motion-fields at test time: ',num2str(norm(reshape(dvfs(:, :,:,:,max(pars.train_indices)+1:end),[],1)-reshape(dvf_approx(:,max(pars.train_indices)+1:end),[],1))/norm(reshape(dvfs(:, :,:,:,max(pars.train_indices)+1:end),[],1))*100),' %'])

