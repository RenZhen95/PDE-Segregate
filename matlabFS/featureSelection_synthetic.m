% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
pyenv(Version="/usr/bin/python3.11");
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

for i=0:49
    dataset_iteration30 = dictionary(datasets_30obs(i));
    ANDOR_30 = dictionary(dataset_iteration30("ANDOR"));
    ANDOR_30_X = double(ANDOR_30('X'));
    ANDOR_30_y = double(ANDOR_30('y'))';

    dataset_iteration50 = dictionary(datasets_50obs(i));
    ANDOR_50 = dictionary(dataset_iteration50("ANDOR"));
    ANDOR_50_X = double(ANDOR_50('X'));
    ANDOR_50_y = double(ANDOR_50('y'))';

    dataset_iteration70 = dictionary(datasets_70obs(i));
    ANDOR_70 = dictionary(dataset_iteration70("ANDOR"));
    ANDOR_70_X = double(ANDOR_70('X'));
    ANDOR_70_y = double(ANDOR_70('y'))';

    % LH-Relief
    tStart_LH_30 = cputime;
    [Weight_LM_30, Theta_LM_30] = LHR(ANDOR_30_X', ANDOR_3s0_y, Para4IRelief);
    % writematrix(Weight_LM, 'ANDOR30_ReliefLM.csv'));
    % tEnd_LH = cputime - tStart_LH;
    % 
    % % Standard I-Relief
    % tStart_I = cputime;
    % [Weight_I, Theta_I] = IMRelief_1(X', y, Para4IRelief);
    % writematrix(Weight_I, strcat(ds_name, '_ReliefI', suffix, '.csv'));
    % tEnd_I = cputime - tStart_I;
end
% % Measuring the elapsed time for both methods
% LHRelief_timefile = fopen("LHRelief_time" + suffix + ".csv", "w");
% IRelief_timefile = fopen("IRelief_time" + suffix + ".csv", "w");

%   % Recording elapsed time
%   fprintf(LHRelief_timefile, ds_name + "," + tEnd_LH + "\n");
%   fprintf(IRelief_timefile, ds_name + "," + tEnd_I + "\n");
% end
% 
% fclose(LHRelief_timefile);
% fclose(IRelief_timefile);