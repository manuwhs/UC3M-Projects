function [ H ] = get_H_matrix( X, W, b, activ_f)
%GET_H_MATRIX Summary of this function goes here
%   Detailed explanation goes here

n_Sa = size(X,1);   % Number of sample vectors
Ni = size(X,2);     % Number of parameters of the vectors = N of Input neurons

    Z = X * W.';     % Activation values of the hidden neurons
    
    ind = ones(1,n_Sa); % Add the biases to all neurons
    BiasMatrix = b(ind,:);      
    Z = Z + BiasMatrix;
    
    H = tanh(Z);     % Output of the hidden layer for all neurons
    
end

