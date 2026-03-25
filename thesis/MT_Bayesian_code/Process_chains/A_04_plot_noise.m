clc; clear all;

try
    load('MT_TD_Chain_Processed.mat');
    fprintf('Plotting MT noise hyperparameter \n')
    image_name = 'MT_noise_image';
    inversion_flag = 0;
catch
    try
        load('DC_TD_Chain_Processed.mat');
        fprintf('Plotting DC noise hyperparameter \n')
        image_name = 'DC_noise_image';
        inversion_flag = 1;
    catch
        load('MT_DC_TD_Chain_Processed.mat');
        fprintf('Plotting MT+DC noise hyperparameter \n')
        image_name = 'MT_noise_image';
        inversion_flag = 2;
    end
end

figure
subplot(2,1,1)
hold on
for j = 1:CData.nChains_atT1
    if inversion_flag == 2
        plot(1:size(sigma,1), sigma(:,j,1), '-k'); % set(gca,'XScale','log'); % axis([0 length(Samples.misfit) 0 10]);
        plot(1:size(sigma,1), sigma(:,j,2), '-r')
    else
        plot(1:size(sigma,1), sigma(:,j), '-')
    end
end
xlabel('# samples')
ylabel('Lambda')

subplot(2,1,2)
hold on
temp = sigma(:,:,1);
histogram(temp(:),'Normalization','pdf')
% temp = sigma(:,:,2);
% histogram(temp(:),'Normalization','pdf')
xlabel('Lambda')
ylabel('p(lambda)')
if inversion_flag == 2
% legend ('MT','DCR')
end
f=gcf;
print(f,image_name,'-djpeg',['-r',num2str(600)]);