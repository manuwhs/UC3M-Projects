
function [Likelihood] = get_glmfit_likhood (B,X,Y)
% Given a set {X,Y} of N pair input/output, it gives you the probability
% of getting that set, given that the probability of getting a Y = 1 is
% P (Y = 1) = sigmoid(1,([1, X(:,i)])*(B))
[n1,n_param] = size(X);
[n2,D] = size(B);

if (n1 ~= n2)
%   exit(0);
end


Likelihood_i = zeros(1,n1);

% Get likelihood probability of every point
for i = 1 : n1
    if (Y(i) == 1)
        Likelihood_i(i) = sigmoid(1,([1, X(i,:)])*(B));
    end
    if (Y(i) == 0)
        Likelihood_i(i) = 1 - sigmoid(1,([1, X(i,:)])*(B));
    end
end

Likelihood = 0;
for i = 1 : n1
    Likelihood = Likelihood + log(Likelihood_i(i));
end


end

