function [ P ] = sigmoid(landa,x )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
    P = 1/(1 + exp(-landa * x));
end

