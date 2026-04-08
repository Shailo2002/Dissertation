function [normalization, reg_term] = estimate_like_norm(...
    unc_proposed, unc_current, sigma, CData)

normalization = 0;  reg_term = 0;
% compute the data normalisation
if strcmpi(CData.inversion_method, 'MT') || strcmpi(CData.inversion_method, 'MT_DC')
    if strcmpi(CData.MT.datatype,'Z')
        % accounting for the change in the sigma for complex data
        temp = log(2*unc_current.MT_Z ./unc_proposed.MT_Z);
        normalization = normalization + sum(temp);
        reg_term = -0.5 * ((sigma(1) - 1)/0.2)^2; 
    elseif strcmpi(CData.MT.datatype,'app')
        temp = log(unc_current.MT_appres ./unc_proposed.MT_appres);
        normalization = normalization + sum(temp);
        reg_term = -0.5 * ((sigma(1) - 1)/0.2)^2; 
    elseif strcmpi(CData.MT.datatype,'phase')
        temp = log(unc_current.MT_phase ./unc_proposed.MT_phase);
        normalization = normalization + sum(temp);
        reg_term = -0.5 * ((sigma(1) - 1)/0.2)^2; 
    elseif strcmpi(CData.MT.datatype,'app_phase')
        temp = log(unc_current.MT_appres ./unc_proposed.MT_appres);
        normalization = normalization + sum(temp);
        
        temp = log(unc_current.MT_phase ./unc_proposed.MT_phase);
        normalization = normalization + sum(temp);

        reg_term = -0.5 * ((sigma(1) - 1)/0.2)^2; 
        % reg_term = -0.5 * ((sigma(2) - 1)/0.2)^2; 
    end
end

if strcmpi(CData.inversion_method, 'DC') || strcmpi(CData.inversion_method, 'MT_DC')
    temp = log(unc_current.DC ./unc_proposed.DC);
    normalization = normalization + sum(temp);
end
end

