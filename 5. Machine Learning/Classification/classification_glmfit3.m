%%%%%%%%%%%%%%%%%%% LOAD DATA AND TREAT INPUTS %%%%%%%%%%%%

close all;
clear all;
load('datosP1.mat');

%%%%% ADD POLINOMIAL TERMS OF X %%%%%%%%%%55
xTrain = [xTrain polynom_X(xTrain,2) polynom_X(xTrain,3)];
xTest = [xTest polynom_X(xTest,2) polynom_X(xTest,3)];

[Ntrain, D] = size(xTrain);
[Ntest, D] = size(xTest);
n_tr = Ntrain;
n_te = Ntest;

% Convert the outputs into numbers coz they are given as logical.
yTest = double(yTest);
yTrain = double(yTrain);

% Plot the initial data             1)
plot_2D_bindata(xTrain, yTrain);  % Plot training data.

%%%%%%%%%%%%% LOGISTIC REGRESION MODEL %%%%%%%%%%%%%%% 2)
B = glmfit(xTrain,yTrain,'binomial');
% Calculates the weight vector values of the logistic regression

%%%%%%%%%%%%%%%%% CLASSIFY TRAINING AND TESTING %%%%%%%%%%%%%%%%%

%**************** Testing classifier for training ****************
%Ypred_tr = glm_fit_classif_MAP (B,xTrain, Prior_1);
Ypred_tr = glm_fit_classif (B,xTrain);
%*******************Testing classifier for testing**************
%Ypred_te = glm_fit_classif_MAP (B,xTest,Prior_1);
Ypred_te = glm_fit_classif (B,xTest);

%%%%%%%%%%%%%%%%% PLOT CLASSIFIED TRAINING AND TESTING %%%%%%%%%%%%%%%%%
% Plotting the classified training
figure();
plot_2D_bindata(xTrain, Ypred_tr);

% Plotting the classified testing
figure();
plot_2D_bindata(xTest, Ypred_te);

%%%%%%%%%%%%%%%%% LIKELIHOODS %%%%%%%%%%%%%%%%%
% Probability of P(y = 1 | x,w) = sigmoid(1,x*w).
% Probability of P(y = 0 | x,w) = 1 - sigmoid(1,x*w).

% Likelihood of training
Likelihood_Tr = get_glmfit_likhood (B,xTrain,yTrain);
% Likelihood of testing
Likelihood_Te = get_glmfit_likhood (B,xTest,yTest);


%%%%%%%%%%%%%%%%% NUMBER OF ERRORS %%%%%%%%%%%%%%%%%

% Number of errors training
N_errors_tr = get_Nerrors(yTrain,Ypred_tr);
% Number of errors testing
N_errors_te = get_Nerrors(yTest,Ypred_te);

%%%%%%%%%%%%%%%%% Plotting of the MAP of Class 1 %%%%%%%%%%%%%%%%%
% Given a point vector X(i), the probability of that vector belonging to
% the class 1 is P(y = 1 | x,w) = sigmoid(1,x*w). If we have a prior, we
% perform bayesian inference and the posterior probaility of getting that
% vector if we have a class 1 is: 
% P(x = X(i) | y = 1,w) ~= P(y = 1 | x,w) * P(y = 1)
% MAP = (Probability of that vector being a 1) * (Prior of 1)

%Get the prior 
Prior_1 = get_bin_prior (yTrain);

% What we do is generate a grid of values of the two input variables x1
% and x2 and then get the MAP of those values and just plot them.

% Range of x1 and x2 goes from 0 to 1.
accuaricy = 0.05;
N = 1/accuaricy + 1;

MAP = zeros(N,N);
i = 1;
j = 1;
for x1 = 0:accuaricy:1
    for x2 = 0:accuaricy:1
         MAP(i,j) = get_glmfit_MAP_1 ([x1, x2, polynom_X([x1,x2],2),polynom_X([x1,x2],3)],B,Prior_1);
        j = j + 1;
    end
    i = i + 1;
    j = 1;
end
figure();
contourf(MAP.');

% ROC curve
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(yTest,Ypred_te,1);
% The area under the curve is the AUC parameter.

