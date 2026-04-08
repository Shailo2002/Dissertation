function [normalization, CData] = compute_noise(CData, sigma)
if CData.number_of_datasets > 1
    % These are the ongoing uncertainities in the datasets
    CData.MT.err_current = CData.MT.err_proposed;
    CData.DC.err_current = CData.DC.err_proposed;
    
    % Update for both MT and DC
    CData.MT.err_proposed = sigma(1) * CData.MT.err;
    CData.DC.err_proposed = sigma(2) * CData.DC.err;
else
    % one of them will definetly work ....
    if strcmpi(CData.inversion_method, 'MT')
        CData.MT.err_current = CData.MT.err_proposed;
        CData.MT.err_proposed = sigma(1) * CData.MT.err;
    end
    if strcmpi(CData.inversion_method, 'DC')
        CData.DC.err_current = CData.DC.err_proposed;
        CData.DC.err_proposed = sigma(1) * CData.DC.err;
    end
end
% compute the data normalisation
if strcmpi(CData.inversion_method, 'MT')
    normalization = log(CData.MT.err_current./CData.MT.err_proposed);
    normalization = sum(normalization);
elseif strcmpi(CData.inversion_method, 'DC')
    normalization = log(CData.DC.err_current./CData.DC.err_proposed);
    normalization = sum(normalization);
elseif strcmpi(CData.inversion_method, 'MT_DC')
    normalization_1 = log(CData.MT.err_current./CData.MT.err_proposed);
    normalization_2 = log(CData.DC.err_current./CData.DC.err_proposed);
    normalization = sum(normalization_1) +  sum(normalization_2);
end
% if strcmpi(CData.inversion_method, 'MT')
%     
% elseif strcmpi(CData.inversion_method, 'DC')
%     
% elseif strcmpi(CData.inversion_method, 'MT_DC')
%     CData.MT.err_current = CData.MT.err_proposed;
%     CData.DC.err_current = CData.DC.err_proposed;
% end
end
