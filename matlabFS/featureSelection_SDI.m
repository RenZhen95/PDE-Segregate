% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
% pyenv(Version="/usr/bin/python3.11");
pickle = py.importlib.import_module('pickle');
% cd('/home/liaw/repo/PDE-SegregateDatasets/synthetic/Electrical/')
cd('D:\OneDrive\Paper2024\GAMM2024\datasets\synthetic\SDI\')

nClass2_idxs = [16; 43; 70; 97];
nClass3_idxs = [17; 44; 71; 98];
nClass4_idxs = [18; 45; 72; 99];

% processedDatasets_dict = pickle.load(handle);
% handle.close();
% cd('/home/liaw/repo/PDE-Segregate/matlabFS/')

% Parameters of IRELIEF taken from Y.J, Sun
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier = 0;
Para4IRelief.sigma = 0.5;
Para4Relief.KNNpara = 9; % number of the nearest neighbors in KNN
Para4IRelief.Prob = 'yes';
Para4IRelief.NN = [7];

% 4060 features | 3 number of class x 4 iterations per dataset type
Weights_LM_nClass2 = zeros(4060, 4);
Weights_I_nClass2  = zeros(4060, 4);
t_LM = zeros(3, 4);
t_I  = zeros(3, 4);

for i=1:4
    X_nClass2 = readmatrix("X\" + nClass2_idxs(i) + "_X.csv");
    y_nClass2 = readmatrix("y\" + nClass2_idxs(i) + "_y.csv");

    X_nClass3 = readmatrix("X\" + nClass3_idxs(i) + "_X.csv");
    y_nClass3 = readmatrix("y\" + nClass3_idxs(i) + "_y.csv");

    X_nClass4 = readmatrix("X\" + nClass4_idxs(i) + "_X.csv");
    y_nClass4 = readmatrix("y\" + nClass4_idxs(i) + "_y.csv");

    % LH-Relief
    tStart_LH = cputime;
    [Weight_LM, ~] = LHR(X_nClass2', y_nClass2, Para4IRelief);
    Weights_LM(:,i) = Weight_LM;
    tEnd_LH = cputime - tStart_LH;
    t_LM(1, i+1) = tEnd_LH_30;

    % tStart_LH_50 = cputime;
    % [Weight_LM_50, ~] = LHR(X50', y50, Para4IRelief);
    % Weights_LM_50(:,i+1) = Weight_LM_50;
    % tEnd_LH_50 = cputime - tStart_LH_50;
    % t_LM(2, i+1) = tEnd_LH_50;
    % 
    % tStart_LH_70 = cputime;
    % [Weight_LM_70, ~] = LHR(X70', y70, Para4IRelief);
    % Weights_LM_70(:,i+1) = Weight_LM_70;
    % tEnd_LH_70 = cputime - tStart_LH_70;
    % t_LM(3, i+1) = tEnd_LH_70;
    % 
    % % I-Relief
    % tStart_I_30 = cputime;
    % [Weight_I_30, ~] = IMRelief_1(X30', y30, Para4IRelief);
    % Weights_I_30(:,i+1) = Weight_I_30;
    % tEnd_I_30 = cputime - tStart_I_30;
    % t_I(1, i+1) = tEnd_I_30;
    % 
    % tStart_I_50 = cputime;
    % [Weight_I_50, ~] = IMRelief_1(X50', y50, Para4IRelief);
    % Weights_I_50(:,i+1) = Weight_I_50;
    % tEnd_I_50 = cputime - tStart_I_50;
    % t_I(2, i+1) = tEnd_I_50;
    % 
    % tStart_I_70 = cputime;
    % [Weight_I_70, ~] = IMRelief_1(X70', y70, Para4IRelief);
    % Weights_I_70(:,i+1) = Weight_I_70;
    % tEnd_I_70 = cputime - tStart_I_70;
    % t_I(3, i+1) = tEnd_I_70;
end
cd('D:\repo\PDE-Segregate\matlabFS')
% % Save scores and elapsed times
% writematrix(Weights_LM_30, datasetname + "WeightsLM_30.csv");
% writematrix(Weights_LM_50, datasetname + "WeightsLM_50.csv");
% writematrix(Weights_LM_70, datasetname + "WeightsLM_70.csv");
% writematrix(Weights_I_30, datasetname + "WeightsI_30.csv");
% writematrix(Weights_I_50, datasetname + "WeightsI_50.csv");
% writematrix(Weights_I_70, datasetname + "WeightsI_70.csv");
% writematrix(t_LM, datasetname + "_tLM.csv");
% writematrix(t_I, datasetname + "_tI.csv");