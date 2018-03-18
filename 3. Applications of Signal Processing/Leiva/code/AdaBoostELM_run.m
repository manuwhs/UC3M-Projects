function [Fx] = AdaBoostELM_run(X, classifier)
% Runs a pretrained ELM AdaBoost classifiers thourgh X


[Nsamples, Nfeatures] = size(X); 

Nrounds = classifier.Nrounds;

Fx = zeros(Nsamples,1);  % Output of the whole system for every sample

for m = 1:Nrounds
    disp(sprintf('Round %d', m))
    
    % Weak ELM obtaining 
    
    [Y] = simpleELM_run_boost(X, classifier(m).W, classifier(m).b, classifier(m).beta);
    alpha = classifier(m).alpha;
    % Updating and computing classifier output on training samples
    fm = Y;         % Outputs of the Weak Classifier
    Fx = Fx + alpha*fm;   % update strong classifier
    
end

% % We are going to concatenate all the Weight, bias, betas and alphas to
% % improve computation
% 
% total_W = []
% total_b = []
% total_beta = []
% 
% for m = 1:Nrounds
%     total_W = [total_W, classifier(m).W] 
%     total_b = [total_b, classifier(m).b] 
%     total_beta = [total_beta, classifier(m).total_beta] 
%    
% end
% 

