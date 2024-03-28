% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
% pyenv(Version="/usr/bin/python3.11");
pickle = py.importlib.import_module('pickle');
cd('../experiments/synthetic/')

handle = py.open("cont_synthetic_datasets.pkl", 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd('../../matlabFS/')

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

processedDatasets = dictionary(processedDatasets_dict);
datasets_30obs = dictionary(processedDatasets(30));
datasets_50obs = dictionary(processedDatasets(50));
datasets_70obs = dictionary(processedDatasets(70));

% ANDOR
Weights_LM_30_ANDOR = zeros(100, 50);
Weights_LM_50_ANDOR = zeros(100, 50);
Weights_LM_70_ANDOR = zeros(100, 50);
Weights_I_30_ANDOR = zeros(100, 50);
Weights_I_50_ANDOR = zeros(100, 50);
Weights_I_70_ANDOR = zeros(100, 50);
tANDOR_LM = zeros(3, 50);
tANDOR_I = zeros(3, 50);

% ADDER
Weights_LM_30_ADDER = zeros(100, 50);
Weights_LM_50_ADDER = zeros(100, 50);
Weights_LM_70_ADDER = zeros(100, 50);
Weights_I_30_ADDER = zeros(100, 50);
Weights_I_50_ADDER = zeros(100, 50);
Weights_I_70_ADDER = zeros(100, 50);
tADDER_LM = zeros(3, 50);
tADDER_I = zeros(3, 50);

for i=0:49
    dataset_iteration30 = dictionary(datasets_30obs(i));
    dataset_iteration50 = dictionary(datasets_50obs(i));
    dataset_iteration70 = dictionary(datasets_70obs(i));


    % ANDOR
    ANDOR_30 = dictionary(dataset_iteration30("ANDOR"));
    ANDOR_30_X = double(ANDOR_30('X'));
    ANDOR_30_y = double(ANDOR_30('y'))';

    ANDOR_50 = dictionary(dataset_iteration50("ANDOR"));
    ANDOR_50_X = double(ANDOR_50('X'));
    ANDOR_50_y = double(ANDOR_50('y'))';

    ANDOR_70 = dictionary(dataset_iteration70("ANDOR"));
    ANDOR_70_X = double(ANDOR_70('X'));
    ANDOR_70_y = double(ANDOR_70('y'))';

    % LH-Relief
    tStart_LH_30_ANDOR = cputime;
    [Weight_LM_30_ANDOR, ~] = LHR(ANDOR_30_X', ANDOR_30_y, Para4IRelief);
    Weights_LM_30_ANDOR(:,i+1) = Weight_LM_30_ANDOR;
    tEnd_LH_30_ANDOR = cputime - tStart_LH_30_ANDOR;
    tANDOR_LM(1, i+1) = tEnd_LH_30_ANDOR;

    tStart_LH_50_ANDOR = cputime;
    [Weight_LM_50_ANDOR, ~] = LHR(ANDOR_50_X', ANDOR_50_y, Para4IRelief);
    Weights_LM_50_ANDOR(:,i+1) = Weight_LM_50_ANDOR;
    tEnd_LH_50_ANDOR = cputime - tStart_LH_50_ANDOR;
    tANDOR_LM(2, i+1) = tEnd_LH_50_ANDOR;

    tStart_LH_70_ANDOR = cputime;
    [Weight_LM_70_ANDOR, ~] = LHR(ANDOR_70_X', ANDOR_70_y, Para4IRelief);
    Weights_LM_70_ANDOR(:,i+1) = Weight_LM_70_ANDOR;
    tEnd_LH_70_ANDOR = cputime - tStart_LH_70_ANDOR;
    tANDOR_LM(3, i+1) = tEnd_LH_70_ANDOR;

    % I-Relief
    tStart_I_30_ANDOR = cputime;
    [Weight_I_30_ANDOR, ~] = IMRelief_1(ANDOR_30_X', ANDOR_30_y, Para4IRelief);
    Weights_I_30_ANDOR(:,i+1) = Weight_I_30_ANDOR;
    tEnd_I_30_ANDOR = cputime - tStart_I_30_ANDOR;
    tANDOR_I(1, i+1) = tEnd_I_30_ANDOR;
    
    tStart_I_50_ANDOR = cputime;
    [Weight_I_50_ANDOR, ~] = IMRelief_1(ANDOR_50_X', ANDOR_50_y, Para4IRelief);
    Weights_I_50_ANDOR(:,i+1) = Weight_I_50_ANDOR;
    tEnd_I_50_ANDOR = cputime - tStart_I_50_ANDOR;
    tANDOR_I(2, i+1) = tEnd_I_50_ANDOR;

    tStart_I_70_ANDOR = cputime;
    [Weight_I_70_ANDOR, ~] = IMRelief_1(ANDOR_70_X', ANDOR_70_y, Para4IRelief);
    Weights_I_70_ANDOR(:,i+1) = Weight_I_70_ANDOR;
    tEnd_I_70_ANDOR = cputime - tStart_I_70_ANDOR;
    tANDOR_I(3, i+1) = tEnd_I_70_ANDOR;


    % ADDER
    ADDER_30 = dictionary(dataset_iteration30("ADDER"));
    ADDER_30_X = double(ADDER_30('X'));
    ADDER_30_y = double(ADDER_30('y'))';

    ADDER_50 = dictionary(dataset_iteration50("ADDER"));
    ADDER_50_X = double(ADDER_50('X'));
    ADDER_50_y = double(ADDER_50('y'))';

    ADDER_70 = dictionary(dataset_iteration70("ADDER"));
    ADDER_70_X = double(ADDER_70('X'));
    ADDER_70_y = double(ADDER_70('y'))';

    % LH-Relief
    tStart_LH_30_ADDER = cputime;
    [Weight_LM_30_ADDER, ~] = LHR(ADDER_30_X', ADDER_30_y, Para4IRelief);
    Weights_LM_30_ADDER(:,i+1) = Weight_LM_30_ADDER;
    tEnd_LH_30_ADDER = cputime - tStart_LH_30_ADDER;
    tADDER_LM(1, i+1) = tEnd_LH_30_ADDER;

    tStart_LH_50_ADDER = cputime;
    [Weight_LM_50_ADDER, ~] = LHR(ADDER_50_X', ADDER_50_y, Para4IRelief);
    Weights_LM_50_ADDER(:,i+1) = Weight_LM_50_ADDER;
    tEnd_LH_50_ADDER = cputime - tStart_LH_50_ADDER;
    tADDER_LM(2, i+1) = tEnd_LH_50_ADDER;

    tStart_LH_70_ADDER = cputime;
    [Weight_LM_70_ADDER, ~] = LHR(ADDER_70_X', ADDER_70_y, Para4IRelief);
    Weights_LM_70_ADDER(:,i+1) = Weight_LM_70_ADDER;
    tEnd_LH_70_ADDER = cputime - tStart_LH_70_ADDER;
    tADDER_LM(3, i+1) = tEnd_LH_70_ADDER;

    % I-Relief
    tStart_I_30_ADDER = cputime;
    [Weight_I_30_ADDER, ~] = IMRelief_1(ADDER_30_X', ADDER_30_y, Para4IRelief);
    Weights_I_30_ADDER(:,i+1) = Weight_I_30_ADDER;
    tEnd_I_30_ADDER = cputime - tStart_I_30_ADDER;
    tADDER_I(1, i+1) = tEnd_I_30_ADDER;
    
    tStart_I_50_ADDER = cputime;
    [Weight_I_50_ADDER, ~] = IMRelief_1(ADDER_50_X', ADDER_50_y, Para4IRelief);
    Weights_I_50_ADDER(:,i+1) = Weight_I_50_ADDER;
    tEnd_I_50_ADDER = cputime - tStart_I_50_ADDER;
    tADDER_I(2, i+1) = tEnd_I_50_ADDER;

    tStart_I_70_ADDER = cputime;
    [Weight_I_70_ADDER, ~] = IMRelief_1(ADDER_70_X', ADDER_70_y, Para4IRelief);
    Weights_I_70_ADDER(:,i+1) = Weight_I_70_ADDER;
    tEnd_I_70_ADDER = cputime - tStart_I_70_ADDER;
    tADDER_I(3, i+1) = tEnd_I_70_ADDER;
end

% Save scores and elapsed times
% ANDOR
writematrix(Weights_LM_30_ANDOR);
writematrix(Weights_LM_50_ANDOR);
writematrix(Weights_LM_70_ANDOR);
writematrix(Weights_I_30_ANDOR);
writematrix(Weights_I_50_ANDOR);
writematrix(Weights_I_70_ANDOR);
writematrix(tANDOR_LM);
writematrix(tANDOR_I);

% ADDER
writematrix(Weights_LM_30_ADDER);
writematrix(Weights_LM_50_ADDER);
writematrix(Weights_LM_70_ADDER);
writematrix(Weights_I_30_ADDER);
writematrix(Weights_I_50_ADDER);
writematrix(Weights_I_70_ADDER);
writematrix(tADDER_LM);
writematrix(tADDER_I);
