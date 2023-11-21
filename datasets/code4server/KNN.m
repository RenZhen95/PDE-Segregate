function best_accuracy = KNN(X1,Xlabels,K,CVO,method)
        [ns,nf] = size(X1);
switch method
    case 'cv'
        for K_idx = 1:length(K)
            for fold_idx = 1:CVO.NumTestSets
                train_id = find(CVO.training(fold_idx));
                test_id = find(CVO.test(fold_idx));
                train_sample = X1(train_id,:);
                train_label = Xlabels(train_id);
                test_label = Xlabels(test_id);
                test_sample = X1(test_id,:);
                predicted = knnclassify(test_sample, train_sample, train_label,5);
                acc(fold_idx) = sum(predicted==test_label)/length(test_label);
            end
            accuracy (K_idx) = mean(acc);
        end
    case 'loo'
        ind = 1:ns;
        for K_idx = 1:length(K)
            for  fi = 1:ns
                train_id = setdiff(ind,i);;
                test_id = fi;
                train_sample = X1(train_id,:);
                train_label = Xlabels(train_id);
                test_label = Xlabels(test_id);
                test_sample = X1(test_id,:);
                pred = knnclassify(test_sample, train_sample, train_label,K(K_idx));
                tp(fi)  = pred==test_label;
            end
            accuracy (K_idx) = sum(tp)/ns;
        end
end

[best_accuracy,idx4k] = max(accuracy(:));
K_best(1) = K(idx4k)