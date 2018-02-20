function [W,b,beta,Y] = simpleELM_train_boost(X, T, Nh, D)

% This function perform a single ELM training for classification boosting
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

n_Tr = size(X,1);   % Number of training vectors
Ni = size(X,2);     % Number of parameters of the vectors = N of Input neurons

%%%%%%%%% INITIALIZATION  %%%%%%%%%

W = rand(Nh,Ni)*2-1;
b = rand(1,Nh)*2-1;


%%%%%%%%% BETA OBTAINING  %%%%%%%%%

H = get_H_matrix( X, W, b, 'tanh');

Hat = diag(D);
Hinv_D = inv((H.')*Hat*H)*( H.')*Hat;


beta = Hinv_D * T;        % Get the beta by resulution of LS

%%%%%%%%% Output OBTAINING  %%%%%%%%%

Y = H * beta;
Y = tanh(Y);

clear H;

 

