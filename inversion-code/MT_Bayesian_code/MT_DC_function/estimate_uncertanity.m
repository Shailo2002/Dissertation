function uncertanity = estimate_uncertanity(sigma, CData)

% compute the data normalisation
if strcmpi(CData.inversion_method, 'MT')
    if strcmpi(CData.MT.datatype,'Z')
        uncertanity.MT_Z = sigma(1) * CData.MT.err_Z;
    elseif strcmpi(CData.MT.datatype,'app')
        uncertanity.MT_appres = sigma(1) * CData.MT.err_appres;
    elseif strcmpi(CData.MT.datatype,'phase')
        uncertanity.MT_phase = sigma(1) * CData.MT.err_phase;
    elseif strcmpi(CData.MT.datatype,'app_phase')
        uncertanity.MT_appres = sigma(1) * CData.MT.err_appres;
        uncertanity.MT_phase = sigma(2) * CData.MT.err_phase;
    end
elseif strcmpi(CData.inversion_method, 'DC')
    uncertanity.DC = sigma(1) * CData.DC.err_appres;
elseif strcmpi(CData.inversion_method, 'MT_DC')
    if strcmpi(CData.MT.datatype,'Z')
        uncertanity.MT_Z = sigma(1) * CData.MT.err_Z;
    elseif strcmpi(CData.MT.datatype,'app')
        uncertanity.MT_appres = sigma(1) * CData.MT.err_appres;
    elseif strcmpi(CData.MT.datatype,'phase')
        uncertanity.MT_phase = sigma(1) * CData.MT.err_phase;
    elseif strcmpi(CData.MT.datatype,'app_phase')
        uncertanity.MT_appres = sigma(1) * CData.MT.err_appres;
        uncertanity.MT_phase = sigma(2) * CData.MT.err_phase;
    end
    if strcmpi(CData.MT.datatype,'Z')
        uncertanity.DC = sigma(2) * CData.DC.err_appres;
    else
        uncertanity.DC = sigma(3) * CData.DC.err_appres;
    end
end
end