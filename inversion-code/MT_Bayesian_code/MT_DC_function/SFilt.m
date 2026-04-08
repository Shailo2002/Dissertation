function r = SFilt(AB2, thick, rho)
% computes the Schlumberger resistivity
% usage:
%    SFilt(AB/2, thick, rho)
% modified to work with log data and log model.

J1 = -5 ;
J2 = 13;
S = -0.14452175;
DY = .48052648;
F = [0.00097112 -0.00102152 0.00906965 0.01404316 0.09012000 ...
    0.30171582 0.99627084 1.36908320 -2.99681171 1.65463068 ...
    -0.59399277 0.22329813 -0.10119309 0.05186135 -0.02748647 ...
    0.01384932 -0.00599074 0.00190463 -0.00032160];

n = length(AB2);
r = zeros(n,1);
for j = 1:n
    off = log(AB2(j)) + S + DY*(1-J1);
    r(j) = 0;
    for i = 1:J2-J1+1
        off = off - DY;
        lam = 1/exp(off);
        r(j) = r(j) + KRTrans(lam, thick, rho)*F(i);
    end
    r(j) = log10(r(j));
end
end

function trans = KRTrans(lam, thick, rho)
% computes the Koefoed resistivity transform
% usage:
%     trans = KRTrans(lam, thick, rho)
% modified to work with log resistivities as imput

n = length(rho);
trans = 10^rho(n);
for i = n-1:-1:1
    tlt = tanh(thick(i)*lam);
    trans = (trans + 10^rho(i)*tlt)/(1+trans*tlt/10^rho(i));
end
end
