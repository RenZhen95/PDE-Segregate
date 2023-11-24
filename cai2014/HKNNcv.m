function [best_accuracy K_best  lambda_best ] = HKNNcv(data, label, K, lambda,CVO)

[ns, nf] = size(data);
for lambda_idx = 1:length(lambda)
    fold_accuracy = zeros(1,CVO.NumTestSets);
    for fold_idx = 1:CVO.NumTestSets
        train_id = find(CVO.training(fold_idx));
        test_id = find(CVO.test(fold_idx));
        train_sample = data(train_id,:);
        train_label = label(train_id);
        test_label = label(test_id);
        predicted_k = zeros(length(K),length(test_id));
        for length_test = 1: length(test_id)
            test_sample = data(test_id(length_test),:);
            for K_idx = 1:length(K)
                predicted_k(K_idx,length_test) = HKNNclass(test_sample,train_sample,train_label,K(K_idx),lambda(lambda_idx));
            end
        end
        for K_idx= 1:length(K)
            predicted4k = predicted_k(K_idx,:)';
            fold_accuracy4k(K_idx,fold_idx) = sum(predicted4k == test_label)/length(test_label);
        end
        predicted_k =[];predicted_D =[];
    end
    accuracy4k(lambda_idx,1:length(K)) = mean(fold_accuracy4k,2)*100;
end

[best_accuracy(1),idx4k] = max(accuracy4k(:)/100);
[lambda_idx4k,K_idx4k] = ind2sub(size(accuracy4k),idx4k);
K_best(1) = K(K_idx4k); lambda_best(1) = lambda(lambda_idx4k);

