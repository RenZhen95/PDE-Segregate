% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
pyenv(Version="/usr/bin/python3.11");
pickle = py.importlib.import_module('pickle');
cd('/home/liaw/repo/PDE-SegregateDatasets/real/')

suffix = "noANOVAcutclean";
handle = py.open("processedDatasets_" + suffix + ".pkl", 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd('/home/liaw/repo/PDE-Segregate/matlabFS/')
% cd('D:\repo\PDE-Segregate\matlabFS\')

processedDatasets = struct(processedDatasets_dict);
datasetKeys = fieldnames(processedDatasets);

% Measuring the elapsed time
mRMR_timefile = fopen("mRMR_time" + suffix + ".csv", "w");

for ds_i=1:length(datasetKeys)
  ds_name = datasetKeys{ds_i};

  dataset = struct(processedDatasets.(datasetKeys{ds_i}));

  X = double(dataset.X);
  y = double(dataset.y)';
  if ds_name == "geneExpressionCancerRNA"
      y = y+1;
  end

  % mRMR
  tStart_mRMR = cputime;
  idx_rank = fscmrmr(X, y);
  tEnd_mRMR = cputime - tStart_mRMR;

  % Recording elapsed time
  fprintf(mRMR_timefile, ds_name + "," + tEnd_mRMR + "\n");
  % Write out ranks
  writematrix(idx_rank, strcat(ds_name, '_mRMR', suffix, '.csv'));
end

fclose(mRMR_timefile);