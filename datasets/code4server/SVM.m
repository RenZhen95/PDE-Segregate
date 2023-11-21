function bestcv = SVM(X,Xlabels,method)
%switch between LOOCV and fold CV scheme of SVM for binary class
%
X = DataStandard(X);
 [bestcv,bestc,bestg] = svmeval(X,Xlabels);
 switch method  
     case 'cv'
     return
     case 'loo' %% conduct LOOCV
 cmd = ['-t 0 -c ', num2str(bestc), ' -g ', num2str(bestg)];
 id = 1:length(Xlabels);
for idx = 1:length(Xlabels)
    train_id = setdiff(id,idx);
     train_sample = X(train_id,:);
    train_label = Xlabels(train_id);
    test_label = Xlabels(idx);
    test_sample = X(idx,:);
      % [nsv alpha bias] =svc(train_sample,train_label,kernel,kerneloption,C);
      %predicted = svcoutput(train_sample,train_label,test_sample,kernel,kerneloption,alpha,bias,0);
      model =  svmtrain(train_label, train_sample,cmd);
      [~,tp] = svmpredict(test_label, test_sample, model);
     acc(idx) = tp(1);
end
bestcv = mean(acc);
 end

function [bestcv,bestc,bestg] = svmeval(trainData,trainLabel)
%grid search for best parameters
 

%%%%
bestcv=0;
for log2c =-3:3
    for log2g = -4:4
        cmd = ['-v 5 -t 0 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(double(trainLabel), trainData, cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
    end
end