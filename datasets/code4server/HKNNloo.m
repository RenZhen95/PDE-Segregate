function [best_accuracy K_best  lambda_best ] = HKNNloo(data, label, K, lambda)

[ns, nf] = size(data);
ind = 1:ns;
for lambda_idx = 1:length(lambda)
    for K_idx = 1:length(K)
        for fi = 1:ns
            train_id = setdiff(ind,fi);;
            test_id = fi;
            train_sample = data(train_id,:);
            train_label = label(train_id);
            test_label = label(test_id);
            test_sample = data(test_id,:);
            pred = HKNNclass(test_sample,train_sample,train_label,K(K_idx),lambda(lambda_idx));
            tp(fi)  = pred==test_label;
        end
        accuracy (lambda_idx,K_idx) = sum(tp)/ns;
    end
end


[best_accuracy,idx4k] = max(accuracy(:));
[lambda_idx4k,K_idx4k] = ind2sub(size(accuracy),idx4k);
K_best(1) = K(K_idx4k); lambda_best(1) = lambda(lambda_idx4k);

