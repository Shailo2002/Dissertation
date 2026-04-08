function status = save_chain(results_all, is, CData, swapCount, destination_folder)
% In this part we will simply collect the whole chain and later process
% the chain as per our convience (using burn-in and thinning)
% Now please store the data for every nth steps, or at the last step
if mod(is, 100) == 0 || is == CData.nsteps
    for ic = 1:CData.nChains
        if strcmpi(CData.inversion_method,'MT')
            filename = ['MT_TD_Chain_',sprintf('%03d',ic),'.mat'];
        elseif strcmpi(CData.inversion_method,'DC')
            filename = ['DC_TD_Chain_',sprintf('%03d',ic),'.mat'];
        elseif strcmpi(CData.inversion_method,'MT_DC')
            filename = ['MT_DC_TD_Chain_',sprintf('%03d',ic),'.mat'];
        end
        % copy the samples to store them in a single variable
        Samples_Chain = results_all(ic,:);

        save(filename,'Samples_Chain','swapCount','CData', '-v7.3');
        try
            copyfile(filename, destination_folder);
            % delete(filename)
        catch
        end
    end
end
status = 1;
end