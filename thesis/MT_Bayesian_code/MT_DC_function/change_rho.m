function [lerr, model, CData, indx] = change_rho(model, CData)
% select which node has to be updated
indx = randi([1, size(model,1)],1);

temp = model(indx, 2) + randn * CData.sigma_rho;
if temp < CData.min_res_log || temp > CData.max_res_log
    % change not accepted
    lerr = 0;
else
    % change is accepted
    model(indx, 2) = temp;
    lerr = 1;
end
end