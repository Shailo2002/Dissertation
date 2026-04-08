function [] = plotModel1D(x)% plotModel1D(b,l,x)

z = x.z;
r = x.rho;

for i=1:length(r)
   k = 2*i-1;
   rr(k) = r(i);
   rr(k+1) = r(i);
end
for i=1:length(z)-1
    k = 2*i - 1;
   zz(k) = z(i);
   zz(k+1) = z(i+1);
end
zz(k+2) = z(end);
zz(k+3) = 2*z(end);

plot(rr,zz,'--k','linewidth',2)
set (gca,'ydir','reverse')
hold on
end