function [lerr, sigma] = change_noise(sigma, CData)
lerr = 1;
if strcmpi(CData.inversion_method, 'MT') || strcmpi(CData.inversion_method, 'MT_DC')
    if strcmpi(CData.MT.datatype,'Z') 
        temp = sigma(1) + randn * CData.sigma_noise*diff(CData.sigma_Z)/100;
        if temp < CData.sigma_Z(1) || temp > CData.sigma_Z(2)
            lerr = 0;
            return
        else
           sigma(1) = temp; 
        end
    elseif strcmpi(CData.MT.datatype,'app')
        temp = sigma(1) + randn * CData.sigma_noise*diff(CData.sigma_app_res)/100;
        if temp < CData.sigma_app_res(1) || temp > CData.sigma_app_res(2)
            lerr = 0;
            return
        else
           sigma(1) = temp; 
        end
    elseif strcmpi(CData.MT.datatype,'phase')
        temp = sigma(2) + randn * CData.sigma_noise*diff(CData.sigma_phase)/100;
        if temp < CData.sigma_phase(1) || temp > CData.sigma_phase(2)
            lerr = 0;
            return
        else
           sigma(2) = temp; 
        end
    elseif strcmpi(CData.MT.datatype,'app_phase')
        temp = sigma(1) + randn * CData.sigma_noise*diff(CData.sigma_app_res)/100;
        if temp < CData.sigma_app_res(1) || temp > CData.sigma_app_res(2)
            lerr = 0;
            return
        else
           sigma(1) = temp; 
        end
        temp = sigma(2) + randn * CData.sigma_noise*diff(CData.sigma_phase)/100;
        if temp < CData.sigma_phase(1) || temp > CData.sigma_phase(2)
            lerr = 0;
            return
        else
           sigma(2) = temp; 
        end
    end
end

if strcmpi(CData.inversion_method, 'DC')
    temp = sigma(1) + randn * CData.sigma_noise*diff(CData.sigma_app_res)/100;
    if temp < CData.sigma_app_res(1) || temp > CData.sigma_app_res(2)
        lerr = 0;
        return
    else
        sigma(1) = temp;
    end
end

if strcmpi(CData.inversion_method, 'MT_DC')
    temp = sigma(3) + randn * CData.sigma_noise*diff(CData.sigma_app_res)/100;
    if temp < CData.sigma_app_res(1) || temp > CData.sigma_app_res(2)
        lerr = 0;
        return
    else
        sigma(3) = temp;
    end
end
end

