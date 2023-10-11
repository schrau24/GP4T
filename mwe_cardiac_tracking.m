%% init

clear all; close all;

gpml_path = '/nfs/arch11/researchData/USER/nhutting/code/gpml-matlab-master/';

restoredefaultpath;

disp('+Adding GPML toolbox to search path')
addpath(genpath(gpml_path))
run([gpml_path,'startup.m']);




base_path = get_data_dir(matlab.desktop.editor.getActiveFilename);
cd(base_path);
addpath(genpath(base_path));

export_path = [base_path,'exports/'];


%% EXPERIMENT 3: cardiac tracking based on projections - preparations
% NOTE1: the input in this case is 352-dimensional (2 times 176-length spokes), with 990 spokes (timepoints) in total.
% NOTE2: this GP is the INTERPOLATION version as described above; our unseen samples are somewhere 'inbetween' the high-dimensional input samples. 
% NOTE3: when the unseen inputs deviate too much from the samples in the training set the performance will decrease rapidly and the uncertainty will explode.
% NOTE3: because the input is so high-dimensional we'd be wise to use the ARD kernel.

% some settings
filter_targets = 1;
filter_width = 5;
rec_resolution = 1.7; % reconstruction resolution.
data_path = [base_path,'/data/cardiac_cine_2d/data.mat'];

% load data
load(data_path);


% visualize data [drag with left mouse button down]
slicer5d(data);
figure;plot(demean(targets*rec_resolution));legend('Left (-) --> Right (+) displacement', 'Foot (-) --> Head (+) displacement');title('Displacement in mm')


% make projections
projections_FH = squeeze(sum(data,2));
projections_LR = squeeze(sum(data,1));
inputs = [projections_LR.',projections_FH.'];


% filter targets [optional, but takes away some spurious jumps]
if filter_targets
    targets = medfilt1(targets,filter_width,[],1);
end

% plot all training data together
figure;
subplot(3,1,1);
imagesc(projections_LR);colormap gray; axis image; yticks('');title('Input #1 - Projection onto LR axis over time');
subplot(3,1,2);
imagesc(projections_FH);colormap gray; axis image; yticks('');title('Input #2 - Projection onto FH axis over time');
subplot(3,1,3);title('Training targets');
plot(demean(targets*rec_resolution));legend('#1', '#2');title('Target outputs')
set_figure_fullscreen;



%% EXPERIMENT 3: cardiac tracking - GP fit

close all;

% GP parameters
pars=[];
pars.Ntrain=round(size(inputs,1)*0.5);
pars.train_indices = 1:pars.Ntrain;
pars.train_noise_std = 0;
pars.visualize=1;
pars.demean_targets = 1;
pars.GP.covfunc = {@covMaternard,3};
pars.GP.Niter = 300;
pars.GP.hyp=struct('mean', [], 'cov', zeros(1,size(inputs,2)+1), 'lik',0);


pars 
ntargets = size(targets,2);
for i=1:ntargets
    pars.GP.noise_std=max(max(abs(demean(targets(:,i))))*2*.04,1e-5);
    [hyp_opt{i},yhat(:,i),y_uncertainties(:,i),means(i)]=TrainGaussianProcess( inputs , targets(:,i) , pars);
end


%% EXPERIMENT 3: cardiac tracking - plot results on images

% visualization settings
playback_delay = 0.003;
vis_inds = 1:100;


f=figure();
set_figure_fullscreen ;

clearvars imgs cm

% loop over vis indices
jj=1;
for i=[vis_inds]
    cla('reset')
    
    imagesc(data(:,:,:,i));
    axis off; axis image;
    set_background_black;

    colormap gray;
    hold on; 
    plot((yhat(i,1)+means(1)),-(yhat(i,2)+means(2)),'m.','MarkerSize',30);
    hold on;
    plot(targets(i,1),-targets(i,2),'blue.','MarkerSize',30);

    l=legend('Prediction by GP','Reference target','FontSize',20,'Fontweight','bold');
    set(l,'color','none');
    set(l,'textcolor','white')
    set(l,'EdgeColor','none')
    CenterLegend
    pause(playback_delay);
    hold off;
    
    
    [imgs{jj},cm{jj}]=to_cells(gcf,0);

    jj=jj+1;

end
    

% .. and write a GIF
disp('Writing gif...')
CellsToGif(imgs(1:end),cm(1:end),playback_delay*3,[export_path,'/tracking_result.gif'])




