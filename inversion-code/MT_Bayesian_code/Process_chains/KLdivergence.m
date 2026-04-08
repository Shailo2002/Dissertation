%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%    KL divergence function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [KLd] = KLdivergence(P,Q)

if(length(P) ~= length(Q) )
    fprintf('length of input arrays must be the same\n')
    keyboard
    return
end

KLd = 0.0;

for j=1:length(P)
    if( P(j) > 0 && Q(j) > 0 )
        KLd = KLd + P(j)*(log(P(j)) - log(Q(j)));
    end
end

end