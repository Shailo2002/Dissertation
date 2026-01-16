function [lerr, model, CData, indx] = cell_move(model, CData)
% if lerr becomes false then we will not accept the
% desired change as mandated by the birth layer
% select which node has to be moved
model_copy = model;
indx = randi([2, size(model,1)],1);

z = model(indx,1) + CData.sigma_loc_z * randn * (CData.max_z-CData.min_z)/100;

if (z < CData.min_z) || (z > CData.max_z)
    lerr = 0;
else
    model(indx,1) = z;
    temp1 = model(1,1:2);     % top fixed node
    temp2 = model(2:end,1:2); % all the last nodes
    % now i needd to sort according to the rows
    [temp2(:,1), ind] = sort(temp2(:,1));
    temp2(:,2) = temp2(ind,2);
    model = [temp1; temp2];   % final model
    % model = sortrows(model);
    lerr = 1;
end

% check if the layer is too thin
if CData.logdomain == true
    temp = [0; 10.^(model(2:end,1)); 10^CData.max_z];
else
    temp = [0; model(2:end,1); CData.max_z];
end
if min(diff(temp)) < CData.minumum_layer_thickness 
    % undo the change made to the model
    lerr = 0; 
    model =  model_copy;
    return
end
end
