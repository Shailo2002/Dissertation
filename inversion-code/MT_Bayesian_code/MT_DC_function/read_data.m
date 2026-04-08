function CData = read_data(MT_datafile, DC_datafile, CData)

% load the MT dataset
if strcmpi(CData.inversion_method,'MT') || strcmpi(CData.inversion_method,'MT_DC')
    % load the data file in .mat format
    % load('MT_data.mat','MT'); CData.MT = MT;
    data = load(MT_datafile);
    CData.MT.period = data(:,1);
    if strcmpi(CData.MT.datatype,'Z')
        CData.MT.dobs_Z = complex(data(:,2), data(:,3));
        CData.MT.err_Z = data(:,4);
        CData.MT.ndata = 2*length(CData.MT.period);
        fprintf('%s \n','MT Impedance data loaded ');
    else
        % convert dataset to the logscale
        if strcmpi(CData.MT.datatype,'app')
            CData.MT.dobs_appres = data(:,2);
            CData.MT.err_appres = data(:,3);
            CData.MT.err_appres = 0.4343 * CData.MT.err_appres ./ CData.MT.dobs_appres;
            CData.MT.dobs_appres = log10(CData.MT.dobs_appres);
            CData.MT.ndata = length(CData.MT.period);
            fprintf('%s \n','MT Apparent resistivity data loaded ');
        elseif strcmpi(CData.MT.datatype,'phase')
            CData.MT.dobs_phase = data(:,2);
            CData.MT.err_phase = data(:,3);
            CData.MT.ndata = length(CData.MT.period);
            fprintf('%s \n','MT Phase data loaded ');
        elseif strcmpi(CData.MT.datatype,'app_phase')
            CData.MT.dobs_appres = data(:,2);
            CData.MT.err_appres = data(:,3);
            CData.MT.err_appres = 0.4343 * CData.MT.err_appres ./ CData.MT.dobs_appres;
            CData.MT.dobs_appres = log10(CData.MT.dobs_appres);
            CData.MT.dobs_phase = data(:,4);
            CData.MT.err_phase = data(:,5);
            CData.MT.ndata = 2*length(CData.MT.period);
            fprintf('%s \n','MT Apparent resistivity and phase data loaded ');
        end
    end
    fprintf('%s %d \n','Total number of periods are ', length(CData.MT.period));
end

if strcmpi(CData.inversion_method,'DC') || strcmpi(CData.inversion_method,'MT_DC')
    data = load(DC_datafile);
    %     data(:,3) = 0.05 * data(:,2);
    %idx = data(:,3)./data(:,2) < 0.1;
    %data(idx,3) = 0.1*data(idx,2);
    % omit two noisy data points
    % range = [1:25 28:31];
    %     range = 1:24;
    CData.DC.dobs_appres = log10(data(:,2));
    CData.DC.err_appres = 0.4343*data(:,3)./data(:,2);
    CData.DC.OA = data(:,1); % electrode spacing, meters
    fprintf('%s \n','DC Apparent resistivity data loaded ');
    fprintf('%s %d \n','Total number of data points are ', length(CData.DC.dobs_appres));
end
end