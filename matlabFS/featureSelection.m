% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
pyenv(Version="/usr/bin/python3.11");
pickle = py.importlib.import_module('pickle');
cd('../datasets/')

suffix = "noANOVAcutclean";
handle = py.open("processedDatasets_" + suffix + ".pkl", 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd('../matlabFS/')

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

processedDatasets = struct(processedDatasets_dict);
datasetKeys = fieldnames(processedDatasets);

% Measuring the elapsed time for both methods
LHRelief_timefile = fopen("LHRelief_time" + suffix + ".csv", "w");
IRelief_timefile = fopen("IRelief_time" + suffix + ".csv", "w");

for ds_i=1:length(datasetKeys)
  ds_name = datasetKeys{ds_i};

  dataset = struct(processedDatasets.(datasetKeys{ds_i}));
  
  X = double(dataset.X);
  y = double(dataset.y)';
  if ds_name == "geneExpressionCancerRNA"
      y = y+1;
  end

  % LH-Relief
  tStart_LH = cputime;
  [Weight_LM, Theta_LM] = LHR(X', y, Para4IRelief);
  writematrix(Weight_LM, strcat(ds_name, '_ReliefLM', suffix, '.csv'));
  tEnd_LH = cputime - tStart_LH;

  % Standard I-Relief
  tStart_I = cputime;
  [Weight_I, Theta_I] = IMRelief_1(X', y, Para4IRelief);
  writematrix(Weight_I, strcat(ds_name, '_ReliefI', suffix, '.csv'));
  tEnd_I = cputime - tStart_I;

  % Recording elapsed time
  fprintf(LHRelief_timefile, ds_name + "," + tEnd_LH + "\n");
  fprintf(IRelief_timefile, ds_name + "," + tEnd_I + "\n");
end

fclose(LHRelief_timefile);
fclose(IRelief_timefile);