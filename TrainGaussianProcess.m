
function [hyp_opt,Ypred,Vpred,mean_data]=TrainGaussianProcess(x,y,pars)
    % Inputs:
    %   x: Input data:              -   [x] = Ntime x M, for a total of Ntime N_Input_Samples-dimensional inputs.
    %   y: Targets:                 -   [y] = Ntime x N_GP_components, for a total of Ntime N_GP_components-dimensional targets. 
    %                                       NOTE: one GP is currently fit for each output; each GP will then define a mapping: x[t] --> y_i[t].
    %
    %   pars:
    %       pars.train_indices      -   The indices the train the GP on [< Ntime]
    %       pars.Ntrain             -   Number of training samples to use from the provided inputs 'x'.
    %                                       NOTE 1: only used whenever pars.train_indices is not provided.
    %                                       NOTE 2: If shuffle_data is off, the program uses the first pars.Ntrain samples as training, otherwise it first shuffles the data and then takes the first pars.Ntrain.
    %       pars.shuffle_data       -   Randomly select pars.Ntrain train indices from the input.
    %                                       NOTE: only used whenever pars.train_indices is not provided.
    %       pars.train_noise_std    -   Optimize for input data noise [0/1]. 
    %                                       NOTE: Usually the test performance is better when this is not optimized [0], because this optimization is prone to overfitting. When not trained, it will use the value in pars.GP.noise_std.
    %       pars.demean_targets     -   Demean targets [0/1]; substracts the mean from 'y' before processing.
    %                                       NOTE: Currently does not add the mean back after inference, but this can be done externally by adding the 'mean_data' output.
    %       pars.normalize_inputs   -   Normalize inputs between -1 and 1 [0/1] before processing.
    %       pars.visualize          -   Visualize training and inference process results [0/1].
    %       pars.GP                 -   GP-specific parameters
    %           pars.GP.meanfunc    -   Mean function
    %           pars.GP.likfunc     -   Likelihood function
    %           pars.GP.covfunc     -   Covariance function
    %           pars.GP.noise_std   -   Noise standard deviation initialization in targets 'y'
    %           pars.GP.hyp         -   Struct with initial values
    %           pars.GP.inf         -   Inference method
    %           pars.GP.Niter       -   Number of maximum likelihood estimation (MLE) iterations

    % Outputs
    %   hyp_opt:        GP hyperparameters
    %   mean_data:      mean of 'y' over first dimension, to be used later 
    %                   correct for offset in predictions
    %   Ypred:          The GP predictions on ALL inputs x, i.e. the GP posterior mean.
    %   Vpred:          The GP prediction VARIANCE on ALL inputs y, i.e. the GP posterior std SQUARED.
    
    
% load GP library by Rasmussen
% addpath('/nfs/rtsan02/userdata/home/asbrizzi/Desktop/otherPrograms'); 
% addpath(genpath('/nfs/rtsan02/userdata/home/asbrizzi/Desktop/otherPrograms/gpml-matlab-v4.1-2017-10-19'));

if nargin<3
    pars=[];
end

if ~isfield(pars,'GP')
        pars.GP=[];
end

    pars.GP = set_default(pars.GP,'meanfunc',[]);              % empty: not necessary use a mean function
    pars.GP = set_default(pars.GP,'likfunc',@likGauss);        % Gaussian likelihood
    pars.GP = set_default(pars.GP,'covfunc',@covSEiso);        % Squared Exponental covariance function
    pars.GP = set_default(pars.GP,'noise_std',5e-3);           
    pars.GP = set_default(pars.GP,'hyp',struct('mean', [], 'cov', [0 0], 'lik', log(pars.GP.noise_std)));
    pars.GP = set_default(pars.GP,'inf',@infGaussLik);
    pars.GP = set_default(pars.GP,'Niter',300);

    pars = set_default(pars,'Ntrain',length(x)/2);    
    pars = set_default(pars,'demean_targets',1);
    pars = set_default(pars,'normalize_inputs',1);
    pars = set_default(pars,'train_noise_std',0);
    pars = set_default(pars,'shuffle_data',0);
    if pars.shuffle_data
        pars = set_default(pars,'train_indices',sort(randperm(size(x,1),pars.Ntrain),'ascend'));
    else
        pars = set_default(pars,'train_indices',1:pars.Ntrain);
    end
    pars = set_default(pars,'visualize',0);
    
if pars.Ntrain > min(size(x,1),size(y,1))
    warning('Too many training samples, setting to maximum:');
    min(size(x,1),size(y,1))
    pars.Ntrain = min(size(x,1),size(y,1));
end


% set internal parameters
Nt              = size(y,1);     % number of timepoints
M               = size(y,2);     % number of components


% center the targets
if pars.demean_targets
    mean_data = mean(y(pars.train_indices,:),1); % store mean over the first dimension to correct the GP outputs if you want to correct the data back again afterwards.
    y       = y-mean_data;
else
    mean_data = zeros(1,size(y,2));
end


% normalize inputs
if pars.normalize_inputs
    lb = -1;
    ub = 1; % lower and upperbound to normalize inputs to.
    mins = min(x(pars.train_indices,:),[],1); % only compute the extreme values over the training indices; the rest could in theory not be available
    maxs = max(x(pars.train_indices,:),[],1); % only compute the extreme values over the training indices; the rest could in theory not be available
    for j = 1:size(x,2)
        if (maxs(j)-mins(j))~=0
            x(:,j) = (x(:,j)-mins(j))/(maxs(j)-mins(j))*(ub-lb)+lb; % normalize all values based on the extreme over train indices
        else
            x(:,j) = x(:,j).^0; %avoid division by 0 in case of constant input!
        end
    end
end


% extract the train data
xtrain = x(pars.train_indices,:);
Ytrain = y(pars.train_indices,:);



if pars.train_noise_std==0
    if numel(pars.GP.inf)>1
        pars.GP.inf{3}.lik = {@priorDelta}; % the prior is a delta peak, i.e. we are 100% sure that the initial noise value is the correct value.
    else
        prior.lik = {@priorDelta};
        pars.GP.inf = {@infPrior,@infGaussLik,prior};
        
    end
end


% perform MLE, once per output dimension
tic;
for m = 1:M
    hyp_opt{m} = minimize(pars.GP.hyp, @gp, -pars.GP.Niter, pars.GP.inf, pars.GP.meanfunc, pars.GP.covfunc, pars.GP.likfunc, xtrain, Ytrain(:,m));
end
training_time = toc;
disp(['Training done in ',num2str(training_time), ' seconds']);



% in case you want to manually analyze/change the resulting hyperparameters, do something as below:
% hyp_opt{M}=hyp_opt{1};
% hyp_opt{1}.cov(1)=-2;%hyp_opt{1}.cov(1)/10;
% hyp_opt{1}.cov(2)=1;
% hyp_opt{1}.cov=[0.01 -0.97];
% hyp_opt{1}.lik=hyp_opt{1}.lik*1.7;
% hyp_opt{1}.cov
% pars.GP.hyp.lik=pars.GP.hyp.lik*1e3;



% Compute the inference on 'x'. Note that this takes the hyp_opt from the MLE as input.
% Initialize the GP inference outputs:
Ypred = zeros(Nt,M);Vpred = Ypred;
tic
for m = 1:M  % loop over all output dimensions
    % compute the inference on 
    [Ypred(:,m), Vpred(:,m)] = gp(hyp_opt{m}, pars.GP.inf, pars.GP.meanfunc, pars.GP.covfunc, pars.GP.likfunc, xtrain, Ytrain(:,m), x);
end
inference_time=toc;
disp(['Inference done in ',num2str(inference_time), ' seconds']);



%%

if pars.visualize
    % plot results
    figure;plot(sqrt(Vpred));title('GP posterior standard deviation - i.e. uncertainty');
    fplot = zeros(2*Nt,M);
for m=1:M
 
    figure;
    set_figure_fullscreen
    % plot the confidence interval
    fplot(:,m) = [Ypred(:,m)+2*sqrt(Vpred(:,m)); flipdim(Ypred(:,m)-2*sqrt(Vpred(:,m)),1)];
    l1=fill([(1:Nt)'; flipdim((1:Nt)',1)], fplot(:,m), [7 7 7]/10,'DisplayName','95% confidence');l1.EdgeColor=[7 7 7]/10;l1.LineWidth=0.2;
    
    % plot GP posterior mean
    hold on; 
    l2=plot((1:Nt)',Ypred(:,m),'Color',[0.6350 0.0780 0.1840],'Linewidth',2,'DisplayName','GP prediction');hold on; 
    
    % plot reference targets
    l3=plot((1:Nt)', y(:,m), 'k-','DisplayName','Measurement','LineWidth',1.5);grid on;hold on;
    
    % plot training targets
    l4=plot(pars.train_indices',y(pars.train_indices,m),'bo','Markersize',12,'DisplayName','Training target');hold on;
    
    % make-up
    ylim([min(fplot(:,m)),max(fplot(:,m))]*1.2);
    set(gca,'fontsize',15)
    set_background_white;   
    xlabel('Time [# snapshot]','Fontsize',25);
    
    % legends
    legend([l1 l2 l3 l4],'Fontsize',25);
    CenterLegend;
    
end
end



end
