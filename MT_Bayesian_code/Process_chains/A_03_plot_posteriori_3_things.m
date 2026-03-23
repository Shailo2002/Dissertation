clc; clear all;

try
    load('MT_TD_Chain_Stat_info.mat');
    fprintf('Plotting MT chains \n')
    image_name = 'MT_Post_image';
catch
    try
        load('DC_TD_Chain_Stat_info.mat');
        fprintf('Plotting DC chains \n')
        image_name = 'DC_Post_image';
    catch
        load('MT_DC_TD_Chain_Stat_info.mat');
        fprintf('Plotting MT+DC chains \n')
        image_name = 'MT_Post_image';
    end
end

fnt_headg = 14; fnt_ticks = 12;     fnt_label = 12;    marksize = 8;
xwidth = 800; ywidth = 600;

figure1 = figure;
set(gcf,'PaperPositionMode','auto')
set(figure1, 'Position', [0 0 xwidth ywidth])

% S.zPlot = 10.^S.zPlot/1000;

subplot(1,3,1)

h = pcolor(S.rhoPlot,S.zPlot,log10(S.posteriorPDF));
set(h,'EdgeColor','none')
hold on
stairs(S.p5,S.zPlot,'-r','linewidth',1)
stairs(S.p95,S.zPlot,'-r','linewidth',1)
if exist('MT_data.mat','file')
   load('MT_data.mat')
   y.z = [0; cumsum(MT.thicknesses)]; y.z = y.z(1:end-1);
   y.rho = log10(MT.resistivities);
   y.z(1) = -1.;
   y.z(2:end) = (y.z(2:end));
   plotModel1D(y)
end
xlabel('log(\rho) (ohm-m)')
ylabel('log10 Depth (m)')
set(gca,'YDir','reverse','Xtick',-1:1:6)
ylim([0 350000])
set(gca,'FontSize',fnt_ticks)
c = colorbar;
c.Label.String = 'log10 (PDF)';

subplot(1,3,2)
kPDF = nansum(S.kSamples,2)./(nansum(nansum(S.kSamples))*S.dz);
kPrior = mean(kPDF);
kPDF(1) = 0;  % To make the first interface as invalid as its always there
plot(kPDF,S.zPlot(1:end-1),'linewidth',2)
hold on
plot([kPrior kPrior],[S.zPlot(2) S.zPlot(end-1)],'--k')
set(gca,'YDir','reverse')
set(gca,'FontSize',14)
% ylim([S.zPlot(1) S.zPlot(end-1)])
% ylim([log10(100) log10(5000)])
ylim([0 350000])
ylabel('Depth (m)')
xlabel('probability density')

subplot(1,3,3)
histogram(ngrid,'Normalization','pdf')
xlabel('# of subsurface layers')
ylabel('probability density')
set(gca,'FontSize',fnt_ticks)

% set(gcf,'position',[50  250  1092  477])

f=gcf;
print(f,image_name,'-djpeg',['-r',num2str(600)]);

