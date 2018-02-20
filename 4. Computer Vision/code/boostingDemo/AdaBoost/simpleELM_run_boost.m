function [Y] = simpleELM_run_boost(X,W,b,beta)

% This function perform a single ELM training for classification 
% Input:
% X     - Matrix with the x vectors.  The x vectros are rows !
% T     - Target values to be found
% Nh    - Number

% Output: 
% W           - Matrix with the weights of the hidden neurons
% b           - Bias of the hidden neurons
% beta        - Input weights of the output neuron
% o           - Outputs of the system
% TrainingAccuracy      - Training accuracy: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This boosting version is the same as the without boosting one
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_Sa = size(X,1);   % Number of training samples
Ni = size(X,2);     % Number of parameters of the vectors = N of Input neurons

H = get_H_matrix( X, W, b, 'tanh');

Y = H * beta;
% % Y = tanh(Y);

clear H;

