clc; clear all;

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


nChains = CData.nChains;
fprintf('Processing the chains...\n')

like_all = zeros(nChains, CData.nsteps*CData.nsamples);
if inversion_flag == 0
    sigma_all = zeros(nChains, CData.nsteps*CData.nsamples, 2);
    nrms_all = zeros(nChains, CData.nsteps*CData.nsamples, 1);
elseif inversion_flag == 1
    sigma_all = zeros(nChains, CData.nsteps*CData.nsamples, 1);
    nrms_all = zeros(nChains, CData.nsteps*CData.nsamples, 1);
else
    sigma_all = zeros(nChains, CData.nsteps*CData.nsamples,3);
    nrms_all = zeros(nChains, CData.nsteps*CData.nsamples,2);
end

k = 0;
for ic = 1:nChains
    filename = [suffix,sprintf('%03d',ic),'.mat'];
    load(filename);
    steps_in_chain = length(Samples_Chain);
    fprintf('%s %d \n','Processing chain no ..', ic);
    for istep = 1:steps_in_chain
        samp = Samples_Chain(istep);
        samples_in_step = length(samp.misfit);
        samp.temperature = 1;
        if samp.temperature ~= 1
            break
        else
            % This chain has the temperature as 1; will select it
            i1 = (istep-1)*samples_in_step+1; i2 = istep*samples_in_step;  
            like_all(ic,i1:i2) = samp.like;
            if inversion_flag == 0
                % sigma_all(ic,i1:i2,1:2) = samp.sigma;
                nrms_all(ic,i1:i2) = samp.misfit;
            elseif inversion_flag == 1
                % sigma_all(ic,i1:i2) = samp.sigma;
                nrms_all(ic,i1:i2) = samp.misfit;
            else
                % sigma_all(ic,i1:i2,1:3) = samp.sigma;
                nrms_all(ic,i1:i2,1:2) = samp.misfit;
            end
        end
    end
end

figure
subplot(3,1,1)
hold on
[n, m] = size(nrms_all);
for i = 1:n
    plot(1:m, nrms_all(i,:), '-');
end
plot([1 m], [1 1], '--');
set(gca,'XScale','log'); 
axis([0 m -10 10]);

xlabel('# samples')
ylabel('nRMS');

subplot(3,1,2)
hold on
[n, m] = size(nrms_all);
for i = 1:n
    plot(1:m, log10(like_all(i,:)), '-');
end
set(gca,'XScale','log');
xlabel('# samples')
ylabel('log (likelihood)');

subplot(3,1,3)
hold on
[n, m, ~] = size(sigma_all);
if strcmpi(CData.inversion_method,'MT_DC')
    for i = 1:n
        % plot(1:m, sigma_all(i,:,1), '-');
    end
    for i = 1:n
        plot(1:m, sigma_all(i,:,2), '--');
    end
else
    for i = 1:n
        plot(1:m, sigma_all(i,:,1), '-');
        plot(1:m, sigma_all(i,:,2), '-');
    end
end
plot([1 m], [1 1], '--');
set(gca,'XScale','log');
xlabel('# samples')
ylabel('sigma factor');

f=gcf;
print(f,suffix(1:end-1),'-djpeg',['-r',num2str(600)]);