function [LF, nrms] = Residual(dHat, uncertanity, CData)
% Computes residual and arrange 2D/3D deltaR matrix to 1D/2D array
% Data misfit, These delta_R are the in the form of 2D/3D matrix. Also
% compute sum of squares (part of penality function)

np_MT=0; np_DC=0; LF_MT=0; LF_DC=0;

if strcmpi(CData.inversion_method,'MT') || strcmpi(CData.inversion_method,'MT_DC')
    if strcmpi(CData.MT.datatype,'Z')
        res = CData.MT.dobs_Z - dHat.MT.data_Z;
        err = res ./ uncertanity.MT_Z;
        err = conj(err) .* err;
        LF_MT = sum(err(:));    
        % I am making it double so that later in the formula i will divide
        % it by 2 to make the factor 0.5 go away; 
        LF_MT = 2*LF_MT;
    else
        if strcmpi(CData.MT.datatype,'app')
            res = CData.MT.dobs_appres - dHat.MT.data_appres;
            err = res ./ uncertanity.MT_appres;
            err = conj(err) .* err;
            LF_MT = sum(err(:));
        elseif strcmpi(CData.MT.datatype,'phase')
            res = CData.MT.dobs_phase - dHat.MT.data_phase;
            err = res ./ uncertanity.MT_phase;
            err = conj(err) .* err;
            LF_MT = sum(err(:));
        elseif strcmpi(CData.MT.datatype,'app_phase')
            res = CData.MT.dobs_appres - dHat.MT.data_appres;
            err = res ./ uncertanity.MT_appres;
            err = conj(err) .* err;
            LF_MT = sum(err(:));
            res = CData.MT.dobs_phase - dHat.MT.data_phase;
            err = res ./ uncertanity.MT_phase;
            err = conj(err) .* err;
            LF_MT = LF_MT + sum(err(:));
        end
    end 
    np_MT = CData.MT.ndata;
end

if strcmpi(CData.inversion_method,'DC') || strcmpi(CData.inversion_method,'MT_DC')
    res = CData.DC.dobs_appres - dHat.DC.data_appres;
    err = res ./ uncertanity.DC;
    err = err .* err;
    LF_DC = sum(err(:));   np_DC = length(CData.DC.dobs_appres);   
end

% combined likelihood for both MT and the DC part
if strcmpi(CData.inversion_method,'MT')
    if strcmpi(CData.MT.datatype,'Z')
        LF = LF_MT;
    else
        LF = 0.5 * LF_MT;
    end
    nrms(1) = sqrt(LF_MT/np_MT);
elseif strcmpi(CData.inversion_method,'DC')
    LF = 0.5 * LF_DC;
    nrms(1) = sqrt(LF_DC/np_DC);
elseif strcmpi(CData.inversion_method,'MT_DC')
    if strcmpi(CData.MT.datatype,'Z')
        LF = 0.5 * LF_DC + LF_MT;
    else
        LF = 0.5 * LF_DC + 0.5 * LF_MT;
    end
    nrms(1,1) = sqrt(LF_MT/np_MT);
    nrms(1,2) = sqrt(LF_DC/np_DC);
    nrms(1,3) = sqrt((LF_MT+LF_DC)/(np_MT+np_DC));
end
end