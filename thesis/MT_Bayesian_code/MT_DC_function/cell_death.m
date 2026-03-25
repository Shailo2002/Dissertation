function [prob, lerr, updated_model, CData] = cell_death(model, CData)
% if lerr becomes false then we will not accept the
% desired change as mandated by the birth layer

if size(model,1) <= CData.minnodes
    % no node can be deleted as we are already at the minimum limit
    lerr = 0;
    prob = 0;
    updated_model = model;
    return
end
% select which node has to be removed, except the first one
indx = randi([2, size(model,1)],1);

% resistivity value that has to be removed
rho = model(indx,2);   
% the removed part is being filled by the resistivity from the top grid
rho1 = model(indx-1,2);

% Update the model
updated_model = [model(1:indx-1,:); model(indx+1:end,:)];

if(CData.kernel == 0)   % Gaussian kernel
    % Find the nearest grid and the distance to that point
    % indx = find(updated_model(:,1)<=  model(indx,1));  indx = indx(end);
    
    % I am assuming that the values belong to the nearest point
    % rho1 = updated_model(indx, 2);
    % Find the corresponding value of resistivity
    prob  = log(1/(CData.sigma_rho_birth*sqrt(2*pi))) - (rho1-rho)^2/(2*CData.sigma_rho_birth^2) ...
            +log(CData.max_res_log-CData.min_res_log);
else
    % prior
    prob = 0;
end
lerr = 1;
end
