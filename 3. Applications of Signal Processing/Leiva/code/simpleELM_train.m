function [W,b,beta,Y] = simpleELM_train(X, T, Nh)

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

n_Tr = size(X,1);   % Number of training vectors
Ni = size(X,2);     % Number of parameters of the vectors = N of Input neurons

%%%%%%%%% INITIALIZATION  %%%%%%%%%

W = rand(Nh,Ni)*2-1;
b = rand(1,Nh)*2-1;


%%%%%%%%% BETA OBTAINING  %%%%%%%%%

H = get_H_matrix( X, W, b, 'tanh');
Hinv = pinv(H);          % Get the pseudoinvere of H

% Hinv = inv((H.')*H)*(H.');
% ad = sum(sum( Hinv1 - Hinv))

beta = Hinv * T;        % Get the beta by resulution of LS

%%%%%%%%% Output OBTAINING  %%%%%%%%%


Y = H * beta;
Y = tanh(Y);

clear H;

 