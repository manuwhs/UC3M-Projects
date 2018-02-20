
function [MAP] = get_glmfit_MAP (X,B,Y,Prior_1)
% Given a set {X,Y} of N pair input/output, it gives you the posterior probability
% of getting that set, given that the probability of getting a Y = 1 is
% P (Y = 1) = sigmoid(1,([1, X(:,i)])*(B)) and the probability of getting a
% 1 is P(Y = 1) = Prior_1

Prior_0 = 1 - Prior_1;
%Get likelihood
[n1,n_param] = size(X);
[n2,D] = size(B);

if (n1 ~= n2)
 %   exit(0);
end

Likelihood_i = zeros(1,n1);
MAP_i = zeros(1,n1);

% Get likelihood probability of every point
for i = 1 : n1
    if (Y(i) == 1)
        Likelihood_i(i) = sigmoid(1,([1, X(i,:)])*(B));
    elseif (Y(i) == 0)
        Likelihood_i(i) =(1 - sigmoid(1,([1, X(i,:)])*(B)));
    end
end

% Get posterior probability of every point
for i = 1 : n1
    if (Y(i) == 1)
        MAP_i(i) = (Likelihood_i(i) * Prior_1)/((Likelihood_i(i) * Prior_1)+((1 - Likelihood_i(i)) * Prior_0));
    elseif (Y(i) == 0)
        MAP_i(i) = (Likelihood_i(i) * Prior_0)/((Likelihood_i(i) * Prior_0)+((1 - Likelihood_i(i)) * Prior_1));
    end
end

% Get total MAP probability
MAP = 0;
for i = 1 : n1
    MAP = MAP + log(MAP_i(i));
end

end

