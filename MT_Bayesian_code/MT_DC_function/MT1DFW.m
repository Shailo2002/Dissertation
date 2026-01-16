function[Z,appres,phase] = MT1DFW(resistivities,thicknesses,period)

mu = 4*pi*1E-7;                  % Magnetic Permeability (H/m)
omega = 2*pi./period;
n=length(resistivities);         % Number of Layers
Z = zeros(length(omega),1); appres = Z; phase = Z;
for iperiod = 1:length(omega)
    w = omega(iperiod);
    impedances = zeros(n,1);
    
    Zn = sqrt(sqrt(-1)*w*mu*resistivities(n));              % Zn - Basement Impedance
    impedances(n) = Zn;
    
    for j = n-1:-1:1
        resistivity = resistivities(j);
        thickness = thicknesses(j);
        
        dj = sqrt(sqrt(-1)*(w*mu*(1/resistivity)));         % di - Induction parameter
        
        wj = dj*resistivity;                                % wi - Intrinsic Impedance
        
        ej = exp(-2*thickness*dj);                          % ei - Exponential Factor
        
        belowImpedance = impedances(j + 1);
        rj = (wj - belowImpedance)/(wj + belowImpedance);   % ri - Reflection coeficient
        re = rj*ej;                                         % re - Earth R.C.
        Zj = wj * ((1 - re)/(1 + re));                      % Zi - Layer Impedance
        impedances(j) = Zj;
    end
    Z(iperiod) = impedances(1);
    absZ = abs(Z(iperiod));
    appres(iperiod) = (absZ*absZ)/(mu*w);
    phase(iperiod) = atan2(imag(Z(iperiod)),real(Z(iperiod)))*(180/pi);
end
end
