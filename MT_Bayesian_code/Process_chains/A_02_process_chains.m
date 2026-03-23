clc; clear;

try
    load('MT_TD_Chain_001.mat');
    inversion_flag = 0;
    suffix = 'MT_TD_Chain_';
    image_name = 'MT_convergence_image';
    fprintf('Processing MT chains \n')
catch
    try
        load('DC_TD_Chain_001.mat');
        inversion_flag = 1;
        suffix = 'DC_TD_Chain_';
        image_name = 'MT_convergence_image';
        fprintf('Processing DC chains \n')
    catch
        load('MT_DC_TD_Chain_001.mat');
        inversion_flag = 2;
        suffix = 'MT_DC_TD_Chain_';
        image_name = 'MT_convergence_image';
        fprintf('Processing MT+DC chains \n')
    end
end

nsteps = 20;          step_discard = floor(nsteps*5/10);
chain_thin = 25;       nrms_limit = 20;
% information for binning, make changes here
S.zMin = 0; S.zMax = 350000;    S.dz = 1000;
S.rhoMin = -1;     S.rhoMax = 6;             S.drho = 0.1; % in log10(rho) (ohm-m)

nsamples = (nsteps-step_discard)*CData.nChains_atT1*CData.nsamples/chain_thin;
nsamples_chain = (nsteps-step_discard)*CData.nsamples/chain_thin;
fprintf('%s %d  \n','Total Chains :', CData.nChains_atT1);
fprintf('%s %d  \n','Total Steps :', nsteps);
fprintf('%s %d  \n','Total Samples in a step : ', CData.nsamples);
fprintf('%s %d  \n','Total steps discarded : ', step_discard);
fprintf('%s %d  \n','chain thinning : ', chain_thin);
fprintf('%s %d  \n','Total Samples after burin-in and thinning : ', nsamples);

rho_all = zeros(nsamples,CData.maxnodes);   z_all = rho_all;  
like_all = zeros(nsamples,1);               ngrid = zeros(nsamples,1);      
step = zeros(nsamples,1);
if inversion_flag == 2
    sigma = zeros(nsamples_chain, CData.nChains_atT1,3);
    nrms_all = zeros(nsamples,3);
else
    sigma = zeros(nsamples_chain, CData.nChains_atT1);
    nrms_all = zeros(nsamples,1);
end
k = 0;
for ic = 1:CData.nChains_atT1
    filename = [suffix,sprintf('%03d',ic),'.mat'];
    load(filename);
    fprintf('%s %d \n','Processing chains no ', ic)
    j = 0;
    for istep = 1:nsteps
        samp = Samples_Chain(istep);
        samp.temperature = 1;
        if samp.temperature ~= 1
            break
        else
            % This chain has the temperature as 1; will select it
            if istep > step_discard
                for i = 1:chain_thin:length(samp.step)
                    k = k + 1; j = j + 1;
                    ngrids = samp.ncells(i);
                    if CData.logdomain == true
                        z_all(k,1:ngrids) = [0 10.^samp.model(i,2:ngrids,1)];
                    else
                        z_all(k,1:ngrids) = [0 samp.model(i,2:ngrids,1)];
                    end
                    
                    rho_all(k,1:ngrids) = samp.model(i,1:ngrids,2);
                    nrms_all(k,1) = samp.misfit(i);
                    like_all(k,1) = samp.like(i);
                    if inversion_flag == 2
                        sigma(j,ic,:) = samp.sigma(i,:);
                    else
                        sigma(j,ic) = samp.sigma(i);
                    end
                    ngrid(k,1) = ngrids;
                    step(k,1) = samp.step(i);
                end
            end
        end
    end
end
z_all = z_all(1:k,:);        rho_all = rho_all(1:k,:); 
like_all = like_all(1:k,1);  ngrid = ngrid(1:k,1);        
step = step(1:k,1);
if inversion_flag == 2
    sigma = sigma(1:j,:,:);         nrms_all = nrms_all(1:k,:);  
else
    sigma = sigma(1:j,:);           nrms_all = nrms_all(1:k,1);  
end
indx = find(nrms_all(:,1) < nrms_limit);

z_all = z_all(indx,:);        rho_all = rho_all(indx,:); 
like_all = like_all(indx,1);
ngrid = ngrid(indx,1);        step = step(indx,1);
if inversion_flag == 2
    nrms_all = nrms_all(indx,:);
else
    nrms_all = nrms_all(indx,1);
end
fprintf('%s %d  \n','Total Samples selected', length(indx));
fprintf('Saving to .mat file...\n')
save([suffix,'Processed.mat'],'CData','sigma','z_all','rho_all',...
    'nrms_all','like_all','ngrid','step','-v7.3')

fprintf('Computing various statistical information ...\n')
S.nZbins = ceil(( S.zMax - S.zMin )/S.dz);    % number of depth bins
S.zPlot = S.zMin+S.dz/2+(0:S.nZbins-1)*S.dz;  % Depth axis of our PDF plots (midpoints of depth bins)
% S.zPlot = 10.^S.zPlot;

S.rho = S.rhoMin : S.drho : S.rhoMax;
S.nRho = length(S.rho) - 1;
S.prior = (1/(S.rhoMax-S.rhoMin)) * ones(S.nRho,1);

S.rhoBinEdges = S.rhoMin:S.drho:S.rhoMax;
S.nRhobins = length(S.rhoBinEdges) - 1;
S.rhoPlot = S.rhoMin + S.drho/2 + (0:S.nRhobins-1)*S.drho; % rho axis of our PDF plots (midpoints of rho bins)

S.nSamples = size(ngrid,1);
S.rhoSamples = nan(S.nZbins,S.nSamples);
S.kSamples = nan(S.nZbins,S.nSamples);

iProgress = 1;
for iSample = 1:S.nSamples
    n_grids = ngrid(iSample);   % Number of interfaces
    z = [z_all(iSample,1:n_grids) S.zMax];
    temp = z(2:end,1)-z(1:end-1,1);
    if min(temp) <= 0
        disp('here');
    end
    rho = rho_all(iSample,1:n_grids);
    % set the counting for all the models
    for j = 1:n_grids   % Till the last layer
        iZbin1 = find(S.zPlot >= z(j), 1, 'first');
        iZbin2 = find(S.zPlot < z(j+1), 1, 'last');
        if ~isempty(iZbin1)
            S.rhoSamples(iZbin1:iZbin2, iSample) = rho(j);
            S.kSamples(iZbin1, iSample) = 1;
        end
    end
    if(mod(iSample,floor(iProgress*S.nSamples/10)) == 0 )
        fprintf('%d %% samples processed...\n',floor(iProgress*10))
        iProgress = iProgress + 1;
    end
end
S.kSamples = S.kSamples(1:end-1,:);

S.posteriorPDF = zeros(S.nZbins,S.nRhobins);
S.p5 = zeros(S.nZbins,1);
S.p95 = zeros(S.nZbins,1);
S.KLd = zeros(S.nZbins,1);
S.pmean = zeros(S.nZbins,1);
iProgress = 1;
for iZbin=1:S.nZbins
    a = histcounts(S.rhoSamples(iZbin,:),S.rhoBinEdges,'Normalization','pdf');
    S.posteriorPDF(iZbin,:) = a;
    S.p5(iZbin) = prctile(S.rhoSamples(iZbin,:),5);
    S.p95(iZbin) = prctile(S.rhoSamples(iZbin,:),95);
    S.pmean(iZbin) = mean(S.rhoSamples(iZbin,:));
    S.KLd(iZbin) = KLdivergence(S.posteriorPDF(iZbin,:),S.prior);

    if(mod(iZbin,floor(iProgress*S.nZbins/10)) == 0 )
        fprintf('%d %% complete...\n',floor(iProgress*10))
        iProgress = iProgress + 1;
    end
end
S.KLd(1) = 0;  % To make the first interface as invalid as its always there

S.p5 = [S.p5(1); S.p5(1:end-1)];
S.p95 = [S.p95(1); S.p95(1:end-1)];

save([suffix,'Stat_info.mat'],'S','nrms_all','ngrid','step','-v7.3')