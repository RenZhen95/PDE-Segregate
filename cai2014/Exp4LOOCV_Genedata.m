function   Exp4LOOCV_Genedata
close all
% dataname = {'dlbcl.mat','pros1.mat','gcm.mat',...
%     'pros2.mat','cns.mat','leuk.mat','pros3.mat','colon.mat',...
%     'lung.mat'}
dataname = {"leuk.mat"}
FSSAddress ='BiClassData/'
% specify the install address of LIBSVM and mRMR
% add4svm = 'X:/Matlab/work/libsvm/';
% addpath(genpath(add4svm));

for i=1:length(dataname)
    name = strcat(FSSAddress,dataname{i});
    [X,Y] = myLoadData(name);
    %%% Update the label to be [1..n];
    X = zscore(X);
    Y = update_label(Y);
    % %%%t- test to remove the genes whose p-value <0.05
    X = precessed_by_ttest(X,Y,0.005);
    % %%%% Estimate the prune parameter via 5-fold CV for
    % %%%%HKNN, KNN, SVM;
    % % [CV_LH,CV_IRELIEF,CV_ID4LH,CV_ID4IR] = FSelTest(X,Y,'cv');
    [LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR] = FSelTest(X,Y,'loo');
    % fname = sprintf( strcat('Results4',dataname{i}) );
    % % iSaveX(fname,CV_LH,CV_IRELIEF,CV_ID4LH,CV_ID4IR,LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR);
    % iSaveXLiaw(fname,LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR);
end
 

function Y = update_label(Y)
uy = unique(Y);
miny = min(uy);
if miny==-1
    Y = (Y+1)/2+1;
else
    Y = Y-miny+1;
end

function iSaveX(fname,CV_LH,CV_IRELIEF,CV_ID4LH,CV_ID4IR,LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR)
cd('Results4GeneLocal')
save( fname);
cd ..

function iSaveXLiaw(fname,LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR)
cd('liawResults4')
save( fname);
cd ..


function [X,Y] = myLoadData(name)
persistent loadedData

if isempty(loadedData)
    loadedData = load(name);
end
data = loadedData.data;
X = data(:,2:end);
Y = data(:,1);
