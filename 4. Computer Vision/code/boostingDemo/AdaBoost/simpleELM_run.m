function [Y] = simpleELM_run(X, W,b,beta)

% Usage: OSELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, N0, Block)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = OSELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, N0, Block)
%
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

n_Sa = size(X,1);   % Number of training samples
Ni = size(X,2);     % Number of parameters of the vectors = N of Input neurons

H = get_H_matrix( X, W, b, 'tanh');

Y = H * beta;
clear H;


