% run experiment for sparsity gradient boosing & 
% standard gradient boosting
% by Cameron P.H. Chen @Princeton 
% contact: pohsuan [at] princeton [dot] edu

clear


% set the algorithm options
fprintf('loading options\n')
options.data_name = 'housepower';
options.time = clock;
options.time = [date '-' num2str(options.time(4)) num2str(options.time(5))];
options.input_path = '../data/input/'; 
options.working_path = '../data/working/' ; 
options.output_path = '../data/output/' ; 
options.random_seed = 1;
options.exp_to_run = {  'GB', 'SGB'}; %Gradient Boosting,Sparse GB
options.hypothesis_space = 'step_function';

parameters.sparsity_constarint = 100;

% load the data
fprintf('loading data')
load([ options.input_path options.data_name '_data.mat']);




