function dHat = Fwd_response(model, CData)

% convert all to the linear and the last boundary value
if CData.logdomain == true
    z = [0; 10.^(model(2:end,1)); 10^CData.max_z];
else
    z = [0; model(2:end,1); CData.max_z];
end

% Compute and store the FWD response
if strcmpi(CData.inversion_method,'MT') || strcmpi(CData.inversion_method,'MT_DC')    
    % convert to the desired scale    
    thicknesses = CData.scale*(z(2:end,1) - z(1:end-1,1));
    resistivities = 10.^(model(1:end,2));
    period = CData.MT.period;
    [Z, appres, phase] = MT1DFW(resistivities, thicknesses, period);
    
    if strcmpi(CData.MT.datatype,'Z')
        dHat.MT.data_Z = Z;
    else
        if strcmpi(CData.MT.datatype,'app')
            dHat.MT.data_appres = log10(appres);
        elseif strcmpi(CData.MT.datatype,'phase')
            dHat.MT.data_phase = phase;
        elseif strcmpi(CData.MT.datatype,'app_phase')
            dHat.MT.data_appres = log10(appres);
            dHat.MT.data_phase = phase;
        end
    end
end

if strcmpi(CData.inversion_method,'DC') || strcmpi(CData.inversion_method,'MT_DC')
    % converted to the linear resistivity
%     model(:,2) = 10.^(model(:,2));
%     dpred = dcgsafwd_1(CData.DC, model);
%     dpred = log10(dpred);
    
    AB2 = CData.DC.OA;
    thicknesses = CData.scale*(z(2:end,1) - z(1:end-1,1));
    rho = model(1:end,2);
    
    dpred = DC1DFW(AB2, thicknesses, rho);
    dHat.DC.data_appres = dpred;
    
    if(sum(imag(dHat.DC.data_appres)~=0))
        dHat.DC.data_appres = abs(dHat.DC.data_appres);
    end
end
end