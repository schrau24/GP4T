function [calibrated_std, scaling_factor] = CalibrateUncertainties(input_std, predictions, reference_targets, alpha)
    % Transform qualitative uncertainty predictions to quantitative confidence intervals
    % See: Angelopoulos, Anastasios N., and Stephen Bates. "A gentle introduction to conformal prediction and distribution-free uncertainty quantification." arXiv preprint arXiv:2107.07511 (2021).
    
    if nargin<4
        alpha=0.05;
    end

    cal_scores = abs(predictions - reference_targets)./ (input_std);
    cal_N = size(cal_scores,1);

    scaling_factor = quantile(cal_scores, ceil((1-alpha)*(cal_N+1))/cal_N, 1);
    calibrated_std = input_std*scaling_factor;

    
end 