% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc

pickle = py.importlib.import_module('pickle');
cd('..\datasets\')
handle = py.open('processedDatasets.pkl', 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd('..\matlabFS\')

% Parameters of IRELIEF taken from Y.J, Sun
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier = 0;
Para4IRelief.sigma = 0.5;
Para4Relief.KNNpara = 9; % number of the nearest neighbors in KNN
Para4IRelief.Prob = 'yes';
Para4IRelief.NN = [5:2:20];

processedDatasets = struct(processedDatasets_dict);
datasetKeys = fieldnames(processedDatasets);
for ds_i=1:length(datasetKeys)
  ds_name = datasetKeys{ds_i};
  if ds_name ~= "geneExpressionCancerRNA"
    continue
  end
  dataset = struct(processedDatasets.(datasetKeys{ds_i}));
  X = double(dataset.X);
  y = double(dataset.y)';
  if ds_name == "geneExpressionCancerRNA"
      y = y+1;
  end

  % LH-Relief
  [Weight_LM, Theta_LM] = LHR(X', y, Para4IRelief);
  writematrix(Weight_LM, strcat(ds_name, '_ReliefLM.csv'));

  % Standard I-Relief
  [Weight_I, Theta_I] = IMRelief_1(X', y, Para4IRelief);
  writematrix(Weight_I, strcat(ds_name, '_ReliefI.csv'));
end
