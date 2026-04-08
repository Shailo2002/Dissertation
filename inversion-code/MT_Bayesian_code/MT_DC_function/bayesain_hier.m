function [samples, model_current, like_current, sigma_current]...
    = bayesain_hier(model_current, like_current, sigma_current, ...
    CData, ic, is)

% RandStream.setGlobalStream(randomstream);

% initialise the memory
samples.step = zeros(CData.nsamples,1);
samples.ncells = zeros(CData.nsamples,1);
samples.model = zeros(CData.nsamples,CData.maxnodes,2);
samples.like = zeros(CData.nsamples,1);
if strcmpi(CData.inversion_method,'MT')
    if strcmpi(CData.MT.datatype,'app_phase')
        samples.sigma = zeros(CData.nsamples,2);
    else
        samples.sigma = zeros(CData.nsamples,1);
    end
    samples.misfit = zeros(CData.nsamples,1);
elseif strcmpi(CData.inversion_method,'DC')
    samples.sigma = zeros(CData.nsamples,1);
    samples.misfit = zeros(CData.nsamples,1);
elseif strcmpi(CData.inversion_method,'MT_DC')
    if strcmpi(CData.MT.datatype,'app_phase')
        samples.sigma = zeros(CData.nsamples,3);
    else
        samples.sigma = zeros(CData.nsamples,2);
    end
    samples.misfit = zeros(CData.nsamples,3);
end
acceptance_count = zeros(2,6);
acceptance_all = zeros(11,1);

% make a cpoy of perturbation parameters for use
temperature = CData.temperature(ic);
CData.sigma_rho = CData.sigma_rho(ic);
CData.sigma_rho_birth = CData.sigma_rho_birth(ic);
CData.sigma_rho_delayed = CData.sigma_rho_delayed(ic);

CData.sigma_loc_z = CData.sigma_loc_z(ic);
CData.sigma_loc_z_delayed = CData.sigma_loc_z_delayed(ic);

% Forward responses for the desired case, MT/DC/Both
data_current = Fwd_response(model_current, CData);

% compute the uncertanity
uncertanity_current = estimate_uncertanity(sigma_current, CData);

% Compute likelihood and misfit
[like_current, nrms] = Residual(data_current, uncertanity_current, CData);
 
model_proposed = model_current;
sigma_proposed = sigma_current;

iFM = 0;  iselect = 0; 
while iselect < CData.nsamples
    iFM = iFM + 1;
    
    ptype = proposeType(rand, CData);
    
    if ptype == 1
        % cell birth will take place
        [prob, lerr, model_proposed, CData] = cell_birth(model_current, CData);
    elseif ptype == 2
        % cell death will take place
        [prob, lerr, model_proposed, CData] = cell_death(model_current, CData);
    elseif ptype == 3
        % cell location will move
        [lerr, model_proposed, CData, cell_indx] = cell_move(model_current, CData);
    elseif ptype == 4
        % cell in resistivity value of the node
        [lerr, model_proposed, CData, cell_indx] = change_rho(model_current, CData);
        % write_info(lerr, ptype);
    elseif ptype == 5
        % cell in noise level, right now only the value of lambda
        if CData.log_normal_noise == true
            [lerr, sigma_proposed, alpha_prior_ratio] = change_noise2(sigma_current, CData);
        else
            [lerr, sigma_proposed] = change_noise(sigma_current, CData);
        end
        % [iFM iselect]
    end
    
    if lerr == 1
        % the model is within the bounds
        if ptype <= 4
            % perform fwd modelling
            % Call Forward responses for the desired case, MT/DC/Both
            data_proposed = Fwd_response(model_proposed, CData);
            
            % Compute likelihood and misfit
            [like_proposed, nrms] = Residual(data_proposed, uncertanity_current, CData);
            
            normalization = 0;
        else
            data_proposed = data_current;
            
            % compute the uncertanity
            uncertanity_proposed = estimate_uncertanity(sigma_proposed, CData);
            % Compute likelihood and misfit
            [like_proposed, nrms] = Residual(data_proposed, uncertanity_proposed, CData);

            % i need to modify this to incorporate the 
            [normalization, reg_term] = estimate_like_norm (...
                uncertanity_proposed, uncertanity_current, sigma_proposed, CData);

            if CData.log_normal_noise == true
                like_proposed = like_proposed + reg_term;
            end
        end
    else
        continue
    end
    
    if abs(like_proposed) < CData.eps
        % the likelihood is a very small number hence, discard
        disp('Like is very small... '); pause
        continue
    end
    
    if ptype == 1
        if(CData.kernel==0)
            alpha = min([0,prob + (-like_proposed+like_current+normalization)/temperature]);
        else
            alpha = min([0,(-like_proposed+like_current+normalization)/temperature]);
        end
    elseif ptype == 2
        if(CData.kernel==0)
            alpha = min([0,prob + (-like_proposed+like_current+normalization)/temperature]);
        else
            alpha = min([0,(-like_proposed+like_current+normalization)/temperature]);
        end
    else
        % normalization = 0;
        % move, value change or noise
        if ptype == 5
            if CData.log_normal_noise == true
                alpha = min([0,(-like_proposed+like_current+normalization+log(alpha_prior_ratio))/temperature]);
            else
                alpha = min([0,(-like_proposed+like_current+normalization)/temperature]);
            end
        else
            alpha = min([0,(-like_proposed+like_current+normalization)/temperature]);
        end
    end
    
    if alpha > log(rand)
        % The model has been accepted; update all things
        like_current = like_proposed; 
        if ptype <= 4   % vornoi model was changed
            model_current = model_proposed;
            data_current = data_proposed;
        else
            sigma_current = sigma_proposed;    
        end
        % Store the model after every step
        iselect = iselect + 1;
        nn = size(model_current, 1);
        samples.step(iselect) = ptype;
        samples.ncells(iselect) = nn;
        
        samples.model(iselect,1:nn,1:2) = model_current;
        samples.like(iselect) = like_current;
        samples.misfit(iselect,:) = nrms;
        samples.sigma(iselect,:) = sigma_proposed;
        
        acceptance_count(1,ptype) = acceptance_count(1,ptype) + 1;
    else
        acceptance_count(2,ptype) = acceptance_count(2,ptype) + 1;
    end
end
temp = acceptance_count(1:2,:);
if sum(temp(:)) > 0
    acceptance_all(1) = 100*sum(acceptance_count(1,:))/sum(temp(:));
end
for iptype = 1:5
    if sum(acceptance_count(1:2,iptype)) > 0
        acceptance_all(iptype+1) = 100*acceptance_count(1,iptype)/sum(acceptance_count(1:2,iptype));
    end
end
samples.acceptance_all = acceptance_all;
samples.acceptance_count = acceptance_count;

% print the infor
fprintf('%s %2d %4d %8d %8d\n','AR [chain step FM FM_call nRMS]',...
    ic, is, CData.nsamples, sum(temp(:)));

if strcmpi(CData.inversion_method,'MT_DC')
    fprintf('%s %8.2f %8.2f %3d\n','   [Accp_rate nRMS Layers]',...
        acceptance_all(1), nrms(3), nn);
else
    fprintf('%s %8.2f %8.2f %3d\n','   [Accp_rate nRMS Layers]',...
        acceptance_all(1), nrms, nn);
end
fprintf('%s %8.2f %8.2f %8.2f %8.2f %8.2f\n','   BDMCN ',acceptance_all(2:6));
end