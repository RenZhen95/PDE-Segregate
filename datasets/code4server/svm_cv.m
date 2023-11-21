function accuracy = svm_cv(X,Y,Best,nfold)
addpath('X:\Matlab\work\WanXb\SVM');
nbclass = max(unique(Y));
[ns, nf] = size(X);
split_assignments = cross_val_split(nfold,ns);
Y= mod(Y,2)*2-1; % change label from [1 2] to [-1 1];
kernel = 'rbf';
fold_accuracy = zeros(1,nfold);

for fold_idx = 1:nfold
    test_id = find(split_assignments(:,fold_idx)==1);
    train_id = find(split_assignments(:,fold_idx)==0);
    train_sample = X(train_id,:);
    train_label = Y(train_id);
    test_sample =  X(test_id,:);
    test_label = Y(test_id);
    [nsv alpha bias] =svc(train_sample,train_label,kernel,Best.kerneloption,Best.C);
    predicted = svcoutput(train_sample,train_label,test_sample,kernel,Best.kerneloption,alpha,bias,0);
    fold_accuracy(fold_idx) = sum(predicted==test_label)/length(test_label);;
    end
accuracy = mean(fold_accuracy)*100;