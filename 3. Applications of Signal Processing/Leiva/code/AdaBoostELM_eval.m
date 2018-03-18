function [Fx, Error_eval] = AdaBoostELM_eval(X,T, classifier)
% Runs a pretrained ELM AdaBoost classifiers thourgh X


[Nsamples, Nfeatures] = size(X); 

Nrounds = classifier.Nrounds;

Fx = zeros(Nsamples,1);  % Output of the whole system for every sample

Error_eval = zeros(1,Nrounds);

for m = 1:Nrounds
    disp(sprintf('Round %d', m))
    
    % Weak ELM obtaining 
    
    [Y] = simpleELM_run_boost(X, classifier(m).W, classifier(m).b, classifier(m).beta);
    alpha = classifier(m).alpha;
    % Updating and computing classifier output on training samples
    fm = Y;         % Outputs of the Weak Classifier
    Fx = Fx + alpha*fm;   % update strong classifier
    
    Error_eval(m) =  get_Pe(Fx,T);
end

