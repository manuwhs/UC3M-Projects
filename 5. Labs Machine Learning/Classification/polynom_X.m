function [ X_poly ] = polynom_X(X,order)
%POLYNOM_X Summary of this function goes here
% We can define the product between polinomials as the convolution 
% between the signals.

[N_samples, N_input] = size(X);
X_poly = zeros(N_samples, (N_input - 1)*order + 1);

for i = 1:N_samples
    aux = X(i,:);
    for o = 1:order-1
        aux = conv(aux,X(i,:));
    end
    X_poly(i,:) = aux;
end
end


