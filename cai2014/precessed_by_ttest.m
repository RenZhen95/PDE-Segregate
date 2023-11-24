function Data = precessed_by_ttest(data,label,sig_level)
% sig_level = 0.05;
% ttest2 return 1 if the null hypothesis is REJECTED at sig_level:
% Data in vectors x and y comes from independent random samples from normal 
% distributions with equal means and equal but unknown variances
[ns,nf]  = size(data);
id = unique(label);
for f = 1:nf
    id1 = find(label==id(1));
    id2 = find(label==id(2));
    H(f) = ttest2(data(id1,f),data(id2,f),sig_level);
end
    % Taking features whose null hypothesis was rejected
    id = find(H);
    Data = data(:,id);
