function [prob, lerr, model, CData] = cell_birth(model, CData)

if size(model,1) >= CData.maxnodes
    % no new poins can be added as we are already at the maximum limit
    lerr = 0;
    prob = 0;
    return
end

for i = 1:1000
    % create the new location of the layer % cc=(LA + (LB-LA) * rand(10,1));
    z = CData.min_z + (CData.max_z-CData.min_z)*rand;

    if CData.logdomain == true
        temp = sort([0; 10.^(model(2:end,1)); 10^z; 10^CData.max_z]);
    else
        temp = sort([0; model(2:end,1); z; CData.max_z]);
    end
    if min(diff(temp)) > CData.minumum_layer_thickness
        break
    end
end
if i == 1000
    % no node was succefully added, break
    lerr = 0;   prob = 0;
    return
end
model(1,1) = 0;

% Now i got the node and i will add it
% Find the nearest neighbour and the distance to that point
[~, indx] = min(abs(model(1:end,1) - z));

% indx = find(model(2:end,1) <=  z);
% if ~isempty(indx)
%     indx = indx(end) + 1;
% else
%     indx = 1;
% end

if(CData.kernel == 0)   % Gaussian kernel
    % Find the resistivity of the nearest point
    rho = model(indx,2);
    rhop = rho + CData.sigma_rho_birth * randn;
    prob  = log(CData.sigma_rho_birth*sqrt(2*pi)) + (rho-rhop)^2/(2*CData.sigma_rho_birth^2) ...
        -log(CData.max_res_log-CData.min_res_log);
    rho = rhop;
else
    % prior
    rho = CData.min_res_log + rand*(CData.max_res_log-CData.min_res_log);
    prob = 0;
end

if(rho < CData.min_res_log || rho > CData.max_res_log)
    lerr = 0;
    prob = 0;
    return
else
    % update the model parameter by putting it back in the main model
    % vector
    model = [model(1:indx,:); z rho; model(indx+1:end,:)];
    lerr = 1;
end
model(1,1) = NaN;
end
