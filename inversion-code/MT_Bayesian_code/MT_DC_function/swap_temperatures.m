function [likelihood, model, sigma, swapCount] = swap_temperatures...
    (likelihood, model, sigma, data, istep, swapCount)

% fid = fopen(data.temperature_swap_file,'a');
if data.jumptype == 0
    % randomly choose a temperature
    % select two chains, one cold and another warm
    indx_1 = randsample(find(data.temperature==1),data.nchain_for_PT);
    indx_2 = randsample(find(data.temperature~=1),data.nchain_for_PT);

    if isempty(indx_2)
        fprintf('%s\n','There are no warm chains');
        return
    end
    
    for i = 1:data.nchain_for_PT
        like_1 = likelihood{indx_1(i)};          like_2 = likelihood{indx_2(i)};
        temp_1 = data.temperature(indx_1(i));    temp_2 = data.temperature(indx_2(i));
        
        alpha_swap = min([0, -like_1/temp_2 + like_2/temp_2 - like_2/temp_1 + like_1/temp_1]);
        
        % Now see if we want to swap
        if log(rand) < alpha_swap
            % let's swap the model and the likelihood
            temp = likelihood{indx_1(i)};  
            likelihood{indx_1(i)} = likelihood{indx_2(i)};
            likelihood{indx_2(i)} = temp;
            
            temp = model{indx_1(i)};  
            model{indx_1(i)} = model{indx_2(i)};
            model{indx_2(i)} = temp;
            
            temp = sigma{indx_1(i)};  
            sigma{indx_1(i)} = sigma{indx_2(i)};
            sigma{indx_2(i)} = temp;
                       
            fprintf('%s %2d %2d %s %4.2f %4.2f\n','PT b/w',[indx_1(i) indx_2(i)],...
               ' and T ',[data.temperature(indx_1(i)) data.temperature(indx_2(i))]);
            % fprintf('%5d %5d %5d \n',[indx_1(i) indx_2(i) istep]);
            swapCount{indx_1(i)}(istep) = 1;
            swapCount{indx_2(i)}(istep) = 1;
        end
    end
elseif data.jumptype == 1
    % massive parallel jumps between various chains
    [indx_1, indx_2] = determinPerm(length(data.temperature));
    
    for i = 1:length(indx_1)
        %twoInts = randperm(nChains,2);        
        like_1 =  likelihood{indx_1(i)};         like_2 = likelihood{indx_2(i)};
        temp_1 = data.temperature(indx_1(i));    temp_2 = data.temperature(indx_2(i));
        
        alpha_swap = min([0, -like_1/temp_2 + like_2/temp_2 - like_2/temp_1 + like_1/temp_1]);
        
        % Now see if we want to swap
        if log(rand) < alpha_swap
            % let's swap the model and the likelihood
            temp = likelihood{indx_1(i)};  
            likelihood{indx_1(i)} = likelihood{indx_2(i)};
            likelihood{indx_2(i)} = temp;
            
            temp = model{indx_1(i)};  
            model{indx_1(i)} = model{indx_2(i)};
            model{indx_2(i)} = temp;
            
            temp = sigma{indx_1(i)};  
            sigma{indx_1(i)} = sigma{indx_2(i)};
            sigma{indx_2(i)} = temp;
            
            temp = uncertainity{indx_1(i)};  
            uncertainity{indx_1(i)} = uncertainity{indx_2(i)};
            uncertainity{indx_2(i)} = temp;
            
            % fprintf('%s %2d %2d %s %4.2f %4.2f %s %2d\n','PT b/w',[indx_1(i) indx_2(i)],...
            %    ' and T ',[data.temperature(indx_1(i)) data.temperature(indx_2(i))],' step ', istep);
            % fprintf(fid, '%5d %5d %5d \n',[indx_1(i) indx_2(i) istep]);
            
            swapCount{indx_1(i)}(i) = 1;
            swapCount{indx_2(i)}(i) = 1;
            
        end
    end
              
elseif data.jumptype == 2
    % choose the nearest temperature

end
% fclose(fid);
end

function [p,q] = determinPerm(n)
%returns index pairs p,q form strictly upper triangular matrix
    k = randperm(n/2*(n-1));
    q = floor(sqrt(8*(k-1) + 1)/2 + 3/2);
    p = k - (q-1).*(q-2)/2;
end