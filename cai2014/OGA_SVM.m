function bestcv = OGA_SVM(X,Y,CVO,method)
%switch between LOOCV and fold CV scheme of SVM for multiclass
%

 [~,~,labels] = unique(Y);   %# labels: 1/2/3
 numLabels = max(labels);
% %data = zscore(meas);              %# scale features
[bestcv,bestc,bestg] = svmeval(X,Y);
switch method 
    case 'cv'
     return
    case  'loo'  %% conduct LOOCV
 cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
 id = 1:length(Y);
for idx = 1:length(Y)
    train_id = setdiff(id,idx);
     train_sample = X(train_id,:);
    train_label = Y(train_id);
    test_label = Y(idx);
    test_sample = X(idx,:);
      % [nsv alpha bias] =svc(train_sample,train_label,kernel,kerneloption,C);
      %predicted = svcoutput(train_sample,train_label,test_sample,kernel,kerneloption,alpha,bias,0);
      model =  svmtrain(train_label, train_sample,cmd);
      [~,tp] = svmpredict(test_label, test_sample, model);
     acc(idx) = tp(1);
end
bestcv = mean(acc);
 end
 
function acc =SVM_predict(trainData,trainLabel,testData,testLabel,numLabels,cmd)

numTest = size(testData,1);
model = cell(numLabels,1);
cmd =[cmd, '-t 2 -b 1'];
for k=1:numLabels
    model{k} = svmtrain(double(trainLabel==k), trainData,cmd);
end

%# get probability estimates of test instances using each model
prob = zeros(numTest,numLabels);
for k=1:numLabels
    [~,~,p] = svmpredict(double(testLabel==k), testData, model{k});
    prob(:,k) = p(:,model{k}.Label==k);    %# probability of class==k
end

%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
 acc  =sum(pred == testLabel) ./ numel(testLabel);  

%C = confusionmat(testLabel, pred) 

function [bestcv,bestc,bestg] = svmeval(trainData,trainLabel)
%grid search for best parameters
[t1,t2] = hist(trainLabel,unique(trainLabel));
[~,id]= max(t1); 
k = t2(id); % 按照样本数量最多的来把数据分为两类。

%%%%
bestcv=0;
for log2c =-1:3
    for log2g = -4:1
        cmd = ['-v 5 -t 2 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(double(trainLabel==k), trainData, cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
    end
end