function [prob,lerr,model_new,CData]=cell_birth(model_current,CData)
lerr=1;
prob=0;
model_new=model_current;
n=size(model_current,1);
if n>=CData.maxnodes
lerr=0;
return
end
zmin=CData.min_z;
zmax=CData.max_z;
znew=zmin+rand*(zmax-zmin);
rnew=CData.min_res_log+rand*(CData.max_res_log-CData.min_res_log);
model_new=[model_current;znew rnew];
model_new=sortrows(model_new,1);
end
