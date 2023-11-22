function [CV4LH,CV4IRELIEF,bestID,id] = FSelTest (X,Y,method)
%%  Compare the permance of I-RELIEF and LHR using classifier
% The coded is implemented based on Y.J, Sun's IRELIEF.

%%%% LH-RELIEF
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier =0;
Para4IRelief.sigma= 0.5;
Para4Relief.KNNpara = 9;% number of the nearest neighbors in KNN
Para4IRelief.Prob= 'yes';
Para4IRelief.NN = [5:2:20];
Sigma4KernelALH = linspace(0.1,2,5);

Lambda = [0.001 0.1 1];;
[Weight_LM,Theta_LM] = LHR(X', Y,Para4IRelief);
KK = [5:30];
tmCV = zeros(length(Para4IRelief.NN ),length(KK),4);
avg = 0.7;

for i =1:length(Para4IRelief.NN)
  for j =1:length(KK)
    K = min([KK(j),size(X,2)]);
    [~,id] = sort(-Weight_LM(:,i)); id =id(1:K);

    % CV = ClassifiersTest(X(:,id),Y,method);
    kClusters = [5:2:15];

    cp = cvpartition(Y,'k',10);
    CV.KNN = KNN(X(:,id), Y, kClusters, cp, method);
    tp_avg= mean(cell2mat(struct2cell(CV)));
    if tp_avg>avg,
      avg = tp_avg;
      CV4LH = CV;
      bestID = id;
      ideal_K = K;
    end
  end
end

% find the best parameter

%%%% Standard IRELIEF
K = ideal_K;
Sigma= [0.1,0.15,0.2,0.3,0.4,0.5,0.8,1,2];
[Weight ,Theta] = IMRelief_1(X', Y,Para4IRelief);
[~,id] = sort(-Weight); id =id(1:K);
CV4IRELIEF = ClassifiersTest(X(:,id),Y,method);
m_IRELIEF = mean(cell2mat(struct2cell(CV4IRELIEF)));
 
