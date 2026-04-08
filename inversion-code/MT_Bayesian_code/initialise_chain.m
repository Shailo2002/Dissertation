function [model_m0, model_LF0, data_sigma] = initialise_chain(CData)

% create an empty array for string all the values
model_m0 = cell(CData.nChains,1);     model_LF0 = cell(CData.nChains,1);
data_sigma = cell(CData.nChains,1);  

for ic = 1:CData.nChains
    % Total number of grids in the model
    ngrid = CData.minnodes + randi([1 2],1);
    % Now create their y- and z- location
    z = CData.min_z + (CData.max_z-CData.min_z)*rand(ngrid,1);
    % i have just talked about the internal nodes only, what about the
    % first fixed node; append empty node at the top and that will be
    % replaced by the 0 on linear scale when computing FWD response
    z(2:length(z)+1) = unique(z);
    z(1) = NaN;

    r = CData.min_res_log + (CData.max_res_log - CData.min_res_log) * rand(ngrid+1,1);
    % r = log10(100)*ones(length(z),1);

    model_m0{ic} = [z r];    model_LF0{ic} = [];
    if strcmpi(CData.inversion_method, 'MT_DC')
        if strcmpi(CData.MT.datatype,'Z')
            data_sigma{ic} = [1 1];
        else
            data_sigma{ic} = [1 1 1];
        end
    elseif strcmpi(CData.inversion_method, 'MT')
        if strcmpi(CData.MT.datatype,'Z')
            data_sigma{ic} = 1;
        else
            data_sigma{ic} = [1 1];
        end       
    elseif strcmpi(CData.inversion_method, 'DC')
        data_sigma{ic} = 1;
    end
end
end