function bestcv = NB(X,Y,cp,method)

 
 switch method  
     case 'cv'
   NBfan = @(Xtr,Ytr,Xte)(NBeval(Xtr,Ytr,Xte));
    CVErr = crossval('mcr',X,Y,'predfun',NBfan,'partition',cp);
     bestcv =1- CVErr;
     case 'loo' %% conduct LOOCV
   id = 1:length(Y);
for idx = 1:length(Y)
    train_id = setdiff(id,idx);
    train_sample = X(train_id,:);
    train_label = Y(train_id);
    test_label = Y(idx);
    test_sample = X(idx,:);
      % [nsv alpha bias] =svc(train_sample,train_label,kernel,kerneloption,C);
      %predicted = svcoutput(train_sample,train_label,test_sample,kernel,kerneloption,alpha,bias,0);
 O = NaiveBayes.fit(train_sample,train_label);
 predicted = O.predict(test_sample);
  acc(idx) = predicted==test_label;
end
bestcv = sum(acc)/length(Y);;
 end
 

function predicted= NBeval(Xtr,Ytr,Xte)
 O = NaiveBayes.fit(Xtr,Ytr);
 predicted = O.predict(Xte);
 
 