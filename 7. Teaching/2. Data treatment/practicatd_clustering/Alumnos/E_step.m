
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluates posterior membership and loglikehood of a GMM model
%
% Input parameters: X - Patterns arranges columnwise (d x n)
%                   model - A struct including the following fields:
%                     * model.c - number of groups
%                     * model.mu - mean vectors (d x model.c)
%                     * model.Sigma - Covariance matrices (d x d x model.c)
%                     * model.P - Prior distribution over classes
%
% Output parameters: posterior - P(w_i|x_k,model), a (model.c x n) matrix
%                    logL - Loglikehood of the model
%
% JAG - Dec 18, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posterior,logL] = E_step(X,model);

n_muestras = size(X,2);
dim = size(X,1);
posterior = zeros(model.c,n_muestras);

for k = 1:model.c
    
    mu = model.mu(:,k);
    Sigma = model.Sigma(:,:,k);
    prior = model.P(k);
    
    posterior(k,:) = mvnpdf(X',mu',Sigma) * prior;
    
end

L = sum(posterior);
posterior = posterior./(ones(model.c,1)*L);
logL = sum(log(L));

