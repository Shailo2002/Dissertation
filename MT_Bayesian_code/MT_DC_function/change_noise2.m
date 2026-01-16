function [lerr, sigma, alpha_prior_ratio] = change_noise2(sigma, CData)

current_sigma = sigma;
alpha_prior_ratio = 0;

lerr = 1;
if strcmpi(CData.inversion_method, 'MT') || strcmpi(CData.inversion_method, 'MT_DC')
    if strcmpi(CData.MT.datatype,'Z') 
        sigma(1) = propose_new_sigma(sigma(1), CData.sigma_noise, CData.sigma_Z);
        alpha_prior_ratio = lognorm_pdf(sigma(1), 1, 0.2) / lognorm_pdf(current_sigma(1), 1, 0.2);
    elseif strcmpi(CData.MT.datatype,'app')
        sigma(1) = propose_new_sigma(sigma(1), CData.sigma_noise, CData.sigma_app_res);
        alpha_prior_ratio = lognorm_pdf(sigma(1), 1, 0.2) / lognorm_pdf(current_sigma(1), 1, 0.2);
    elseif strcmpi(CData.MT.datatype,'phase')
        sigma(1) = propose_new_sigma(sigma(1), CData.sigma_noise, CData.sigma_phase);
        alpha_prior_ratio = lognorm_pdf(sigma(1), 1, 0.2) / lognorm_pdf(current_sigma(1), 1, 0.2);
    elseif strcmpi(CData.MT.datatype,'app_phase')
        sigma(1) = propose_new_sigma(sigma(1), CData.sigma_noise, CData.sigma_app_res);
        sigma(2) = propose_new_sigma(sigma(2), CData.sigma_noise, CData.sigma_phase);
        alpha_prior_ratio = lognorm_pdf(sigma(1), 1, 0.2) / lognorm_pdf(current_sigma(1), 1, 0.2);
        alpha_prior_ratio = alpha_prior_ratio + lognorm_pdf(sigma(2), 1, 0.2) / lognorm_pdf(current_sigma(2), 1, 0.2);
    end
end

if strcmpi(CData.inversion_method, 'DC')
    sigma(1) = propose_new_sigma(sigma(1), CData.sigma_noise, CData.sigma_app_res);
    alpha_prior_ratio = lognorm_pdf(sigma(1), 1, 0.2) / lognorm_pdf(current_sigma(1), 1, 0.2);
end

if strcmpi(CData.inversion_method, 'MT_DC')
    if strcmpi(CData.MT.datatype,'Z') 
        sigma(2) = propose_new_sigma(sigma(2), CData.sigma_noise, CData.sigma_app_res);
        alpha_prior_ratio = alpha_prior_ratio + lognorm_pdf(sigma(2), 1, 0.2) / lognorm_pdf(current_sigma(2), 1, 0.2);
    else
        sigma(3) = propose_new_sigma(sigma(3), CData.sigma_noise, CData.sigma_app_res);
        alpha_prior_ratio = alpha_prior_ratio + lognorm_pdf(sigma(3), 1, 0.2) / lognorm_pdf(current_sigma(3), 1, 0.2);
    end
end
end

function new_sigma = propose_new_sigma(current_sigma, step_size, prior_sigma)
    % step_size = 0.05; % Reduce step size for more stable estimates
    new_sigma = current_sigma * exp(step_size * randn);              % Log-normal proposal
    new_sigma = max(prior_sigma(1), min(prior_sigma(2), new_sigma)); % Keep within bounds
end


% Log-normal probability density function
function p = lognorm_pdf(x, mu, sigma)
    p = (1 / (x * sigma * sqrt(2 * pi))) * exp(-((log(x) - log(mu))^2) / (2 * sigma^2));
end