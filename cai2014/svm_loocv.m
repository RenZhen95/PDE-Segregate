function accuracy = svm_cv(X,Y,Best)
addpath('X:\Matlab\work\WanXb\SVM');
nbclass = max(unique(Y));
[ns, nf] = size(X);
Y= mod(Y,2)*2-1; % change label from [1 2] to [-1 1];
kernel = 'rbf';
set = [1:ns]; 
accuracy = 0;
for i = 1:ns
    test_id = i;
    train_id = setdiff(set,test_id);
    train_sample = X(train_id,:);
    train_label = Y(train_id);
    test_sample =  X(test_id,:);
    test_label = Y(test_id);
    [nsv alpha bias] =svc(train_sample,train_label,kernel,Best.kerneloption,Best.C);
    predicted = svcoutput(train_sample,train_label,test_sample,kernel,Best.kerneloption,alpha,bias,0);
     accuracy  = accuracy+ (predicted==test_label);
    end
accuracy = fold_accuracy/ns*100;