function   Exp4LOOCV_FSS_Genedata
%%% test the feature selection scheme by coupling with five classifiers
% for ease of comparison, selection of optimal genes should be excute first
%by using Exp4LOOCV_Genedata.m
close all
DN = {'dlbcl','pros1','gcm',...
    'pros2','cns','leuk','pros3','colon',...
    'lung'};
FSSAddress ='BiClassData/'
% specify the install address of LIBSVM and mRMR
add4svm = 'X:/Matlab/work/libsvm/';
addpath(genpath(add4svm));
add4mRMR = 'X:/Matlab/work/mRMR_0.9_compiled';
addpath(genpath(add4mRMR));
%%%%%%%%%%%%%%%%%%%%

for i=1: length(DN)
    % LOAD LOOCV results after LHR
    fname = sprintf( strcat('Results4',DN{i}) );
    load(fname);
    %%% LOAD binary test file
    name = strcat(FSSAddress,DN{i});
    [X,Y] = myLoadData(name);
    num_FFS = length(LOO_ID4LH);
    accuracy = zeros(5,12);
    for m = 1:8
        methods = strcat(DN{i}, '_m',num2str(m),'_n30.mat');
        load(strcat(FSSAddress,methods));
        temp = ClassifiersTest(datatran(:,2:num_FFS),datatran(:,1),'loo');
        accuracy(:,m) = cell2mat(struct2cell(temp));
    end
    X1 = precessed_by_ttest(X,Y,0.01);
    [mr_d] = mrmr_mid_d(X1,Y, num_FFS);
    [mr_q]=mrmr_miq_d(X1, Y, num_FFS);
    temp = ClassifiersTest(X1(:,mr_d),Y,'loo');
    accuracy(:,m+1) = cell2mat(struct2cell(temp));
    temp = ClassifiersTest(X1(:,mr_q),Y,'loo');
    clear X1;
    accuracy(:,m+2) = cell2mat(struct2cell(temp));
    accuracy(:,m+3) = cell2mat(struct2cell(LOO_IRELIEF));
    accuracy(:,m+4) = cell2mat(struct2cell(LOO_LH));
    fname = sprintf( strcat('Results4FFS_',DN{i}) );
    save(fname)
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


function [X,Y] = myLoadData(name)
persistent loadedData

if isempty(loadedData)
    loadedData = load(name);
end
data = loadedData.data;
X = data(:,2:end);
Y = data(:,1);

