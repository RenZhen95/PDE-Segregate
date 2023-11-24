function macc = LDAclassify(X1,Xlabels,CVO,method)
[nf,ns] = size(X1);
 
for fold_idx = 1:CVO.NumTestSets
    train_id = find(CVO.training(fold_idx));
    test_id = find(CVO.test(fold_idx));
    train_sample = X1(train_id,:);
    train_label = Xlabels(train_id);
    test_label = Xlabels(test_id);
    test_sample = X1(test_id,:);
    predicted = classify(test_sample, train_sample, train_label,method);
    acc(fold_idx) = sum(predicted==test_label)/length(test_label);
end
macc = mean(acc);