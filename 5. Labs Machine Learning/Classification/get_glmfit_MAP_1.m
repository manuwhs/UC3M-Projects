
function [MAP] = get_glmfit_MAP_1 (xi,B,Prior_1)
% Returns the Porterior probability of that vector for class 1.
% MAP = (Likelihood of that vector being a 1) * (Prior of 1)
% P(x = X(i) | y = 1,w) ~= P(y = 1 | x,w) * P(y = 1)
% P(x = X(i) | y = 1,w) = P(y = 1 | x,w) * P(y = 1) / 
%               (P(y = 1 | x,w) * P(y = 1) + P(y = 0 | x,w) * P(y = 0) )
%Get likelihood of being a 1.
Likelihood_1 = sigmoid(1,[1, xi]*B);

% Get posterior if it is a 1
MAP = Likelihood_1 * Prior_1;

% Normalize.
Likelihood_0 = (1-Likelihood_1);
Prior_0 = (1-Prior_1);
MAP = MAP/((Likelihood_1 * Prior_1) + (Likelihood_0 * Prior_0));

end

