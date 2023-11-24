function Exp4Fermat_Spiral2
% Produce of synthetic data for Fermat Spiral Problem
% and compare the method of LHR and I-RELIEF by using five 
% Classification algorithems.
 
%for testing by svm, please install LIBsvm tool and configure the installed
% address HA to be absolute address
HA = 'X:/';
addpath(genpath(strcat(HA,'/Matlab/work/libsvm/')));
nn = linspace(0,10000,11);
[X,Y]= Fermat_Spiral;
 
for i=1:9%length(nn)
    close all
    NoisyFeature = randn(size(X,1),nn(i));
    X1 = [X NoisyFeature];
    [CV_LH,CV_IRELIEF,CV_ID4LH,CV_ID4IR] = FSelTest(X1,Y,'cv');
    [LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR]  = FSelTest(X1,Y,'loo') ;
 
fname = sprintf( strcat('Results4Spiral2_',num2str(nn(i)),'.mat') );
iSaveX(fname,CV_LH,CV_IRELIEF,CV_ID4LH,CV_ID4IR,LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR);
end

function iSaveX(fname,CV_LH,CV_IRELIEF,CV_ID4LH,CV_ID4IR,LOO_LH,LOO_IRELIEF,LOO_ID4LH,LOO_ID4IR)
  save( fname);
 

