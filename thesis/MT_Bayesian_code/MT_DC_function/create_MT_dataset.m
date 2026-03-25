clc; clear all; close all;

MT.noise_prcnt = 3;        MT.datatype = 'Z';     

MT.scale = 1;

% Let us first compute the forward response here only, later we can read it
layer_top = [0; 10000; 50000];
layer_rho = [100; 10; 1000];

% Minimum period      MAx Period           Number of Periods
a1 = 1;               a2 = 10000;               a3 = 13;

MT.resistivities = layer_rho;
MT.thicknesses = [layer_top(2:end)-layer_top(1:end-1); 1.5*layer_top(end)]; 

MT.botttom_depth = sum(MT.thicknesses)/MT.scale;

T = logspace(log10(a1),log10(a2),a3)';
period = T;
% load('period.mat','period');

% Compute the Forward response for a model
MT.period=period;
[Z,appres,phase] = MT1DFW(MT.resistivities,MT.thicknesses,MT.period);

if strcmpi(MT.datatype,'Z')
    MT.dobs_noise_free = Z;
    MT.ndata = 2*length(Z);
    temp = randn(length(Z),1);
    % Add noise to the data
    Zerr = Z + 0.01*MT.noise_prcnt * temp .* Z;
    MT.dobs = Zerr;
    MT.err = 0.01*abs(MT.dobs);
else
    MT.dobs = [appres; phase];
    MT.ndata = length(MT.dobs);
    MT.err = 0.01*MT.noise_prcnt*abs(MT.dobs);
end

subplot(2,1,1)
loglog(period,appres,'-ro')
xlabel('Periods (s)')
ylabel('Apparent resistivity')
subplot(2,1,2)
semilogx(period,phase,'-ro')
xlabel('Periods (s)')
ylabel('Phase');

% Now estimate appres and phase and the error from the data
omega = 2*pi./period;
mu = 4*pi*1E-7;
for iperiod = 1:length(omega)
    w = omega(iperiod);
    absZ = abs(MT.dobs(iperiod));
    appres(iperiod) = (absZ*absZ)/(mu*w);
    phase(iperiod) = atan2(imag(MT.dobs(iperiod)),real(MT.dobs(iperiod)))*(180/pi);

    % estimate error
    appres_err(iperiod,1) = 2 * MT.dobs(iperiod) .* (MT.err(iperiod) ./ (w * mu));
    phase_err(iperiod,1) = (MT.err(iperiod) /MT.dobs(iperiod)) /pi*180;
end
save('MT_data.mat','MT');

fid = fopen('MT_data.dat','w');
for i = 1:length(period)
   fprintf(fid,'%12.5f %+10.4E %+10.4E %+10.4E %+10.4E\r\n ',...
       period(i), appres(i), appres_err(i), ...
       phase(i), phase_err(i)); 
end

fid = fopen('MT_data_Z.dat','w');
for i = 1:length(period)
   fprintf(fid,'%12.5f %+10.4E %+10.4E %+10.4E\r\n ',period(i), ...
       real(MT.dobs(i)), imag(MT.dobs(i)), MT.err(i)); 
end