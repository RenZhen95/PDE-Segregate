function Data = precessed_by_ttest(data,label,sig_level)
%sig_level = 0.05;
[ns,nf]  = size(data);
id = unique(label);
for f = 1:nf
    id1 = find(label==id(1));
    id2 = find(label==id(2));
    H(f) = ttest2(data(id1,f),data(id2,f),sig_level);
end
    id = find(H);
    Data = data(:,id);
