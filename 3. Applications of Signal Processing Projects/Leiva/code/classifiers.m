% LOGISTIC REGRESION MODEL
B = glmfit(X_tr_pow,Ytrain,'binomial'); % Calculates the weight vector values of the logistic regression

% USE THE LINEAR CLASIFIER

Ypred_te = zeros(1,n_te);
for te_i=1:n_te
    XW = ([1, X_te_pow(te_i,:)])*(B);
    if (XW <= 0) 
        Ypred_te(te_i) = 0;
    else
        Ypred_te(te_i) = 1;
    end
end

% Number of errors
N_errors = get_Pe(Ypred_te, Ytest);
% 
% ROC curve
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Ytest,Ypred_te,1);
% The area under the curve is the AUC parameter.
% SWM MODEL

 SVMStruct = svmtrain(X_tr_pow,Ytrain);
 Ypred_te = svmclassify(SVMStruct,X_te_pow);
 
N_errors = get_Pe(Ypred_te, Ytest);
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Ytest,Ypred_te,1);
