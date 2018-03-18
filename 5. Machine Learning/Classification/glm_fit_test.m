
function [N_errors, AUC, N_errors2 ] = glm_fit_test (X_train,X_test, Y_train, Y_test)

%******************** Machine Learning Applications *************************
%%%%%%%%%%%%%%%%%%%%%%%%%%% LOGISTIC REGRESION MODEL %%%%%%%%%%%%%%%%%%%%%%%%%
[n_tr,n_param1] = size(X_train);
[n_te,n_param2] = size(X_test);

if (n_param1 ~= n_param2)
 %   exit(0);
end

%Training classifier
B = glmfit(X_train,Y_train,'binomial'); % Calculates the weight vector values of the logistic regression

%Testing classifier
Ypred_te = glm_fit_classif (B,X_test);

%Reusults
% Number of errors
N_errors = get_Nerrors(Y_test,Ypred_te);

% ROC curve
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Y_test,Ypred_te,1);
% The area under the curve is the AUC parameter.

Ypred_tr = glm_fit_classif (B,X_train);

%Reusults
% Number of errors
N_errors2 = get_Nerrors(Y_train,Ypred_tr);

end

