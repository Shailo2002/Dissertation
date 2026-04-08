clc; close all; clear all;

% DC.type = 'Schlumberger';     % Electrode DCuration
% DC.OA   = logspace(0,2,21);   % Current electrode spacings
% DC.OM   = [0.1 0.5];            % Half potential electrode spacing

n = 0;
OA   = logspace(0,3,51);
OM   = [0.5];
for k = 1:length(OA)
    for m = 1:length(OM)
        n = n + 1;
        DC.C1(n,1:3) = [-OA(k) 0 0];  % transmitter electrode   C1       P1  P2       C2
        DC.C2(n,1:3) = [OA(k) 0 0];   % transmitter electrode   o--------o---o--------o
        DC.P1(n,1:3) = [-OM(m) 0 0];  % receiver electrode      |    OA    |
        DC.P2(n,1:3) = [OM(m) 0 0];   % receiver electrode             OM| |
        DC.OA(n,1) = OA(k);
        DC.OM(n,1) = OM(m);
        if DC.OA(n) == DC.OM(n)
            DC.skip(n) = 1;
        else
            DC.skip(n) = 0;
            DC.r(n,1:4) = sqrt(sum(([                      ... % calculate distance between electrodes
                DC.P2(n)-DC.C2(n);           ...
                DC.P2(n)-DC.C1(n);           ...
                DC.P1(n)-DC.C2(n);           ...
                DC.P1(n)-DC.C1(n)]).^2,2));
            DC.G_factor(n,1) = 2*pi*(1/DC.r(n,1)- 1/DC.r(n,2)-...
                1/DC.r(n,3)+ 1/DC.r(n,4))^(-1);
        end % if
    end
end

layer_top = [0; 50; 100;];
layer_rho = [100; 1; 1000];

layers = [layer_top layer_rho];
DC.domain    = 'DC';         % Toggles DC calculation
DC.hank_type = 'FHT';        % Choose Fast Hankel Transform using
                                  %       digital filters, 'NHT' chooses
                                  %       numerical integration scheme
DC.FHT_err   = 1.0000e-008;  % Tolerance of Fast Hankel Transform.
DC.Seg_tol = 1e-6;     % Tolerance on each segment of the integration
DC.NHT_tol = 1e-5;     % Tolerance on the sum of the series
DC.Max_seg = 100;      % Maximum number of segments to sum

tic
[appres, result, DC] = dcgsafwd_1(DC, layers);
toc
DC.dobs = appres;
temp = appres + 0.03 * randn(length(appres),1) .* appres;
DC.err = 0.01*temp;

save('DC_data.mat','DC');

tic
r = SFilt(DC.OA, [layer_top(2:end)-layer_top(1:end-1); 1.5*layer_top(end)] , log10(layer_rho));
toc

% loglog([DC.OA],[result.Z].*[result.G_factor],'-bo');
loglog(DC.OA, temp,'-bo');
hold on
loglog(DC.OA, 10.^r,'-r*');

fid = fopen('DC_data.dat','w');
for i = 1:length(DC.dobs)
   fprintf(fid,'%10.2f%10.2f%10.2f\r\n ',DC.OA(i), temp(i), DC.err(i)); 
end

    fclose(fid);