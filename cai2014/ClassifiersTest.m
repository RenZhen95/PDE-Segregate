function accuracy4 = ClassifiersTest(X,Y,method)
% Five Classifiers: LDA, SVM, KNN,NB and HKNN
% For  LDA and NB, you have to install Toolbox of Statistics, Matlab
% version 7.13;
% For SVM, you have to install LIBSVM tool.
K = [5:2:15];
cp = cvpartition(Y,'k',10);
%%%%%%%%%SVM
nClass = length(unique(Y));
% if nClass<3
%     accuracy4.SVM = SVM(X,Y,method)/100;
%     %%%%%%%%%LDA
%     accuracy4.LDA = DAclassify(X,Y,cp,'linear');
%     accuracy4.NB = NB(X,Y,cp,method);
% else
%     % Multiple class
%     accuracy4.SVM = OGA_SVM(X,Y,cp,method)/100;
%     accuracy4.LDA =MLDAclassify(X,Y,cp,method);
% end
%%%%%%%%%KNN
accuracy4.KNN = KNN(X,Y,K,cp,method);

%%%%%%%%% %%%%%%%%%HKNN
%switch method
%    case 'loo'
%        if size(X,2)>3
%            accuracy4.HKNN = HKNNloo(X, Y, K, 10.^[-3:-1]);
%        else
%            accuracy4.HKNN =  accuracy4.KNN;
%        end
%    case 'cv'
%        if size(X,2)>3
%            accuracy4.HKNN = HKNNcv(X, Y, K, 10.^[-3:-1],cp);
%        else
%            accuracy4.HKNN =  accuracy4.KNN;
%        end
end
 
