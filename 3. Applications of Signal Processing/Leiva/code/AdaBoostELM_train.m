function classifier = AdaBoostELM_train(X, T, Nrounds, Nh)
% Perfroms training of AdaBoost using the Extreme Learning Machine as algorithm


[Nsamples, Nfeatures] = size(X); 
Fx = zeros(Nsamples,1);  % Output of the whole system for every sample
w  = ones(Nsamples,1);   % Weight of each sample

classifier.Nrounds = Nrounds;

for m = 1:Nrounds
    w = w / sum(w);
    disp(sprintf('Round %d', m))
    
    % Weak ELM obtaining 
    
    [W,b,beta,Y] = simpleELM_train_boost(X, T, Nh, w);
    
    % Updating and computing classifier output on training samples
    fm = Y;         % Outputs of the Weak Classifier
    Fx = Fx + fm;   % update strong classifier
    
%     disp(size(Y))
%     disp(size(fm))
%     disp(size(w))

    alpha = 1;
%     % Obtain the dehenphasis alpha of the learner given by RealAdaboost
    r = sum(w.*T.*fm);
    alpha = log((1+r)/(1-r))/2;
    
    % Reweight training samples    
    w = w .* exp(-T.*fm*alpha);
    
    % update parameters classifier
    classifier(m).W = W;
    classifier(m).b = b;
    classifier(m).beta = beta;
    classifier(m).alpha = alpha;
    
end


