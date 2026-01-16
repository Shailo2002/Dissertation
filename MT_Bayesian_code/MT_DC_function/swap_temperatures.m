function [likelihood, model, sigma, swapCount] = swap_temperatures...
    (likelihood, model, sigma, data, istep, swapCount)

if data.jumptype == 0

    idx1 = find(data.temperature == 1);
    idx2 = find(data.temperature ~= 1);

    if isempty(idx2)
        return
    end

    indx_1 = idx1(randperm(length(idx1), data.nchain_for_PT));
    indx_2 = idx2(randperm(length(idx2), data.nchain_for_PT));

    for i = 1:data.nchain_for_PT
        like_1 = likelihood{indx_1(i)};
        like_2 = likelihood{indx_2(i)};
        temp_1 = data.temperature(indx_1(i));
        temp_2 = data.temperature(indx_2(i));

        alpha_swap = min([0, -like_1/temp_2 + like_2/temp_2 ...
                             -like_2/temp_1 + like_1/temp_1]);

        if log(rand) < alpha_swap
            t = likelihood{indx_1(i)};
            likelihood{indx_1(i)} = likelihood{indx_2(i)};
            likelihood{indx_2(i)} = t;

            t = model{indx_1(i)};
            model{indx_1(i)} = model{indx_2(i)};
            model{indx_2(i)} = t;

            t = sigma{indx_1(i)};
            sigma{indx_1(i)} = sigma{indx_2(i)};
            sigma{indx_2(i)} = t;

            swapCount{indx_1(i)}(istep) = 1;
            swapCount{indx_2(i)}(istep) = 1;
        end
    end

elseif data.jumptype == 1

    [indx_1, indx_2] = determinPerm(length(data.temperature));

    for i = 1:length(indx_1)
        like_1 = likelihood{indx_1(i)};
        like_2 = likelihood{indx_2(i)};
        temp_1 = data.temperature(indx_1(i));
        temp_2 = data.temperature(indx_2(i));

        alpha_swap = min([0, -like_1/temp_2 + like_2/temp_2 ...
                             -like_2/temp_1 + like_1/temp_1]);

        if log(rand) < alpha_swap
            t = likelihood{indx_1(i)};
            likelihood{indx_1(i)} = likelihood{indx_2(i)};
            likelihood{indx_2(i)} = t;

            t = model{indx_1(i)};
            model{indx_1(i)} = model{indx_2(i)};
            model{indx_2(i)} = t;

            t = sigma{indx_1(i)};
            sigma{indx_1(i)} = sigma{indx_2(i)};
            sigma{indx_2(i)} = t;

            swapCount{indx_1(i)}(istep) = 1;
            swapCount{indx_2(i)}(istep) = 1;
        end
    end

end
end

function [p,q] = determinPerm(n)
k = randperm(n*(n-1)/2);
q = floor((sqrt(8*(k-1)+1)+3)/2);
p = k - (q-1).*(q-2)/2;
end
