function P1clasif;

clear all; close all; format compact

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 1: SYNTHETIC DATA %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0: Generation of synthetic data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Configurable parameters
N  = 400;            % Size of the training set (same for validation and test)
d  = 3;              % Maximum degree of the polynomial terms
% v  = 4*[0.1 1 -3 4 -4 8 -4 -2 2 -4]';   % Coefficients of the true model
% x  = rand(3*N,2);            % Observations
% 
% xx = polyprod(x,d);
% xe = [ones(3*N,1) xx];
% p  = 1./(1+exp(-xe*v));      %
% y  = rand(3*N,1)>1-p;        % Binary labels
% 
% xTrain = x(1:N,:);        yTrain = y(1:N);
% xVal   = x(N+1:2*N,:);    yVal   = y(N+1:2*N);
% xTest  = x(2*N+1:3*N,:);  yTest  = y(2*N+1:3*N);
% 
% save datosP1 xTrain xVal xTest yTrain yVal yTest

load datosP1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1.1: Logistic Regression: A linear 2D case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('1. LOGISTIC REGRESSION. LINEAR CLASSIFIER.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.1.1. Visualize observations over the 2D plane
figure;
plot(xTrain(~yTrain,1),xTrain(~yTrain,2),'o', xTrain( yTrain,1),xTrain( yTrain,2),'+');
axis equal

%print([ 'figura1' ],  '-depsc2'  ) ;

%%%%%%%%%%%%%%%%%%%%
%%% 1.1.2. Model fit
w  = glmfit(xTrain,yTrain,'binomial');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.1.3. Graphical results
xGrid = 0:0.01:1;
[x1,x2] = meshgrid(xGrid);

figure;
q = 1./(1+exp(-w(1)-w(2)*x1-w(3)*x2));
contourf(x1,x2,q,20); colormap gray; hold on      % Gray-level map of the fitted model
title('Estimated logistic regression model. Linear log-odds')
plot(xTrain(~yTrain,1),xTrain(~yTrain,2),'o', xTrain( yTrain,1),xTrain( yTrain,2),'+');
axis equal

%print([ 'figura2' ],  '-depsc2'  ) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.1.4. Compute error rates

% Posterior class probabilities
pTrain = glmval(w,xTrain,'logit');   
pVal   = glmval(w,xVal,'logit');
pTest  = glmval(w,xTest,'logit');

% Decisions
dTrain = pTrain>0.5;
dVal   = pVal  >0.5;
dTest  = pTest >0.5;

% Errors
disp('Error rates:')
eTrain = mean(dTrain~=yTrain)
eVal   = mean(dVal~=yVal)
eTest  = mean(dTest~=yTest)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.1.5. Compute data likelihoods
disp('Likelihoods fort training, validation and test data:')
lTrain = sum(-yTrain.*log(pTrain) - (1-yTrain).*log(1-pTrain))
lVal   = sum(-yVal  .*log(pVal)   - (1-yVal).*log(1-pVal))
lTest  = sum(-yTest .*log(pTest)  - (1-yTest).*log(1-pTest))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1.2: Logistic Regression: polynomial log-odds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('2. LOGISTIC REGRESSION. NON-LINEAR CLASSIFIER.')

%%%%%%%%%%%%%%%%%%%%
%%% 1.2.1. Model fit

%%% Expand data with polynomial components.
d  = 1;                 % Degree of the polynomial


xxTrain = polyprod(xTrain,d);     % See below the code of this function
xxVal = polyprod(xVal,d);     % See below the code of this function
xxTest = polyprod(xTest,d);     % See below the code of this function


%%% Model fit.
w2  = glmfit(xxTrain,yTrain,'binomial');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2,2. Graphical results
xGrid = 0:0.01:1;
[x1,x2] = meshgrid(xGrid);

%%% Posterior map for the estimated model
figure;
q = w2(1)+w2(2)*x1+w2(3)*x2;
%q = q + w2(4)*x1.^2 + w2(5)*x1.*x2    +w2(6)*x2.^2;
%q = q + w2(7)*x1.^3 + w2(8)*x1.^2.*x2 +w2(9)*x1.*x2.^2 +w2(10)*x2.^3;
q = 1./(1+exp(-q));
contourf(x1,x2,q,20); colormap gray; hold on      % Gray-level map of the fitted model
plot(xTrain(~yTrain,1),xTrain(~yTrain,2),'o', xTrain( yTrain,1),xTrain( yTrain,2),'+');
title('Estimated logistic regression model. Linear log-odds, d=1')
axis equal

%print([ 'figura3' ],  '-depsc2'  ) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2.3. Compute error rates

% Posterior class probabilities
pTrain = glmval(w2,xxTrain,'logit');
pVal   = glmval(w2,xxVal,'logit');
pTest  = glmval(w2,xxTest,'logit');

% Decisions
dTrain = pTrain>0.5;
dVal   = pVal  >0.5;
dTest  = pTest >0.5;

%%% Error rates
disp('Training, validation and test error rates for d=1:')
eTrain = mean(dTrain~=yTrain)
eVal   = mean(dVal~=yVal)
eTest  = mean(dTest~=yTest)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2.4. Compute Likelihoods
disp('Likelihoods for training, validation and test data for d=1:')
lTrain = sum(-yTrain.*log(pTrain)-(1-yTrain).*log(1-pTrain))
lVal   = sum(-yVal  .*log(pVal)  -(1-yVal).*log(1-pVal))
lTest  = sum(-yTest .*log(pTest) -(1-yTest).*log(1-pTest))








%%% Expand data with polynomial components.
d  = 2;                 % Degree of the polynomial


xxTrain = polyprod(xTrain,d);     % See below the code of this function
xxVal = polyprod(xVal,d);     % See below the code of this function
xxTest = polyprod(xTest,d);     % See below the code of this function


%%% Model fit.
w2  = glmfit(xxTrain,yTrain,'binomial');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2,2. Graphical results
xGrid = 0:0.01:1;
[x1,x2] = meshgrid(xGrid);

%%% Posterior map for the estimated model
figure;
q = w2(1)+w2(2)*x1+w2(3)*x2+w2(4)*x1.^2+w2(5)*x1.*x2+w2(6)*x2.^2;
%q = q + w2(7)*x1.^3 +w2(8)*x1.^2.*x2 +w2(9)*x1.*x2.^2 +w2(10)*x2.^3;
q = 1./(1+exp(-q));
contourf(x1,x2,q,20); colormap gray; hold on      % Gray-level map of the fitted model
plot(xTrain(~yTrain,1),xTrain(~yTrain,2),'o', xTrain( yTrain,1),xTrain( yTrain,2),'+');
title('Estimated logistic regression model. Linear log-odds, d=2')
axis equal

%print([ 'figura4' ],  '-depsc2'  ) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2.3. Compute error rates

% Posterior class probabilities
pTrain = glmval(w2,xxTrain,'logit');
pVal   = glmval(w2,xxVal,'logit');
pTest  = glmval(w2,xxTest,'logit');

% Decisions
dTrain = pTrain>0.5;
dVal   = pVal  >0.5;
dTest  = pTest >0.5;

%%% Error rates
disp('Training, validation and test error rates for d=2:')
eTrain = mean(dTrain~=yTrain)
eVal   = mean(dVal~=yVal)
eTest  = mean(dTest~=yTest)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2.4. Compute Likelihoods
disp('Likelihoods for training, validation and test data for d=2:')
lTrain = sum(-yTrain.*log(pTrain)-(1-yTrain).*log(1-pTrain))
lVal   = sum(-yVal  .*log(pVal)  -(1-yVal).*log(1-pVal))
lTest  = sum(-yTest .*log(pTest) -(1-yTest).*log(1-pTest))





%%% Expand data with polynomial components.
d  = 3;                 % Degree of the polynomial


xxTrain = polyprod(xTrain,d);     % See below the code of this function
xxVal = polyprod(xVal,d);     % See below the code of this function
xxTest = polyprod(xTest,d);     % See below the code of this function


%%% Model fit.
w2  = glmfit(xxTrain,yTrain,'binomial');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2,2. Graphical results
xGrid = 0:0.01:1;
[x1,x2] = meshgrid(xGrid);

%%% Posterior map for the estimated model
figure;
q = w2(1)+w2(2)*x1+w2(3)*x2+w2(4)*x1.^2+w2(5)*x1.*x2+w2(6)*x2.^2;
q = q + w2(7)*x1.^3 +w2(8)*x1.^2.*x2 +w2(9)*x1.*x2.^2 +w2(10)*x2.^3;
q = 1./(1+exp(-q));
contourf(x1,x2,q,20); colormap gray; hold on      % Gray-level map of the fitted model
plot(xTrain(~yTrain,1),xTrain(~yTrain,2),'o', xTrain( yTrain,1),xTrain( yTrain,2),'+');
title('Estimated logistic regression model. Linear log-odds, d=3')
axis equal

%print([ 'figura5' ],  '-depsc2'  ) ;























% %%% Posterior map for the true model
% %%% This is not requested, but I have included it for comparison.
% figure;
% z = v(1)+v(2)*x1+v(3)*x2+v(4)*x1.^2+v(5)*x1.*x2+v(6)*x2.^2;
% z = z + v(7)*x1.^3 +v(8)*x1.^2.*x2 +v(9)*x1.*x2.^2 +v(10)*x2.^3;
% z = 1./(1+exp(-z));
% contourf(x1,x2,z,20); colormap gray; hold on;     % Gray-level map of the true model
% plot(x(~y,1),x(~y,2),'o', x(y,1),x(y,2),'+');   % Data
% title('True model. Linear log-odds')
% axis equal
% 
% print([ 'figura4' ],  '-depsc2'  ) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2.3. Compute error rates

% Posterior class probabilities
pTrain = glmval(w2,xxTrain,'logit');
pVal   = glmval(w2,xxVal,'logit');
pTest  = glmval(w2,xxTest,'logit');

% Decisions
dTrain = pTrain>0.5;
dVal   = pVal  >0.5;
dTest  = pTest >0.5;

%%% Error rates
disp('Training, validation and test error rates  for d=3:')
eTrain = mean(dTrain~=yTrain)
eVal   = mean(dVal~=yVal)
eTest  = mean(dTest~=yTest)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.2.4. Compute Likelihoods
disp('Likelihoods for training, validation and test data  for d=3:')
lTrain = sum(-yTrain.*log(pTrain)-(1-yTrain).*log(1-pTrain))
lVal   = sum(-yVal  .*log(pVal)  -(1-yVal).*log(1-pVal))
lTest  = sum(-yTest .*log(pTest) -(1-yTest).*log(1-pTest))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1.3: Training a SVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.3.1. Training a SVM

%%% Set parameters
sigma = 0.5;
C     = 0.1;

%%% Train svm and plot decision boundary
figure
svmStruct0 = svmtrain(xTrain,yTrain,     'kernel_function','rbf',...
                      'rbf_sigma',sigma, 'showplot','true',...
                      'boxconstraint',C, 'method','LS');
dVal   = svmclassify(svmStruct0,xVal,  'showplot',false);
dTrain = svmclassify(svmStruct0,xTrain,'showplot',false);

%%% Error rates
eTrain = mean(dTrain~=yTrain)
eVal   = mean(dVal~=yVal)
drawnow

%print([ 'figura6' ],  '-depsc2'  ) ;
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.3.2. Training a SVM

C  = 10.^(-4:0.1:2);
nC = length(C);

eTrain = zeros(nC,1);
eVal   = zeros(nC,1);

svmStruct=cell(1,nC);%{k}

%%% Explore over C
for k=1:nC;
    
    Ck = C(k);

    svmStruct{1,k} = svmtrain(xTrain,yTrain,   'kernel_function','rbf', ...
                         'rbf_sigma',sigma,  'showplot','false', ...
                         'boxconstraint',Ck, 'method','LS');
    
    % Classify the test set using svmclassify
    dVal   = svmclassify(svmStruct{k},xVal,  'showplot',false);
    dTrain = svmclassify(svmStruct{k},xTrain,'showplot',false);

    %%% Error rates
    eTrain(k) = mean(dTrain~=yTrain);
    eVal(k)   = mean(dVal~=yVal);

end

%%% Plor errors versus C
figure
plot(C,eTrain,'.-',C,eVal,'.-');
legend('Train','Validation')
xlabel('C'); ylabel('Error rate')

%print([ 'figura7' ],  '-depsc2'  ) ;

figure
semilogx(C,eTrain,'.-',C,eVal,'.-');
legend('Train','Validation')
xlabel('C'); ylabel('Error rate')
%print([ 'figura8' ],  '-depsc2'  ) ;

%%% Take the best C
[eMin,iMin] = min(eVal)

% Classify the test set using svmclassify
dTest   = svmclassify(svmStruct{iMin},xTest);

%%% Error rates
eTest   = mean(dTest~=yTest);


%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 2: REAL DATA %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the data and select features for classification
load cancer_dataset
x = cancerInputs';
y = cancerTargets(2,:)';

N3 = size(y,1);
N  = fix(N3/3);

xTrain = x(1:N,:);       yTrain = y(1:N);
xVal   = x(N+1:2*N,:);   yVal   = y(N+1:2*N);
xTest  = x(2*N+1:N3,:);  yTest  = y(2*N+1:N3);

nTrain = size(yTrain,1);
nVal   = size(yVal,1);
nTest  = size(yTest,1);


mx     = mean(xTrain); 
stdx   = std(xTrain);
xTrain = (xTrain - ones(nTrain,1)*mx)./(ones(nTrain, 1)*stdx);
xVal   = (xVal   - ones(nVal,1)*mx)  ./(ones(nVal,1)*stdx);
xTest  = (xTest  - ones(nTest,1)*mx) ./(ones(nTest,1)*stdx);


sigma = 0.2:0.05:8;
nS    = length(sigma)
C     = 10.^(-3:0.05:3);
nC    = length(C)
 
%%% Explore over C
eTrain2 = zeros(nS,nC);
eVal2   = zeros(nS,nC);
for j=1:nS;
    for k=1:nC;
        
        sigmaj = sigma(j);
        Ck = C(k);
        
        svmStruct2{j,k} = svmtrain(xTrain,yTrain,   'kernel_function','rbf', ...
            'rbf_sigma',sigmaj,  'showplot','false', ...
            'boxconstraint',Ck, 'method','LS');
        
        % Classify the test set using svmclassify
        dTrain = svmclassify(svmStruct2{j,k},xTrain,'showplot',false);
        dVal   = svmclassify(svmStruct2{j,k},xVal,  'showplot',false);
        
        %%% Error rates
        eTrain2(j,k) = mean(dTrain~=yTrain);
        eVal2(j,k)   = mean(dVal~=yVal);
        
    end
end

%%% Plor errors versus C
figure
[x1,x2] = meshgrid(log10(C), sigma);
contourf(x1,x2,eTrain2,40);
xlabel('log10(C)')
ylabel('\sigma')
%plot(C,eTrain2,'.-',C,eVal2,'.-');
%legend('Train','Validation')
%print([ 'figura9' ],  '-depsc2'  ) ;

figure
[x1,x2] = meshgrid(log10(C), sigma);
contourf(x1,x2,eVal2,40);
xlabel('log10(C)')
ylabel('\sigma')
%semilogx(C,eTrain2,'.-',C,eVal2,'.-');
%legend('Train','Validation')

%print([ 'figura10' ],  '-depsc2'  ) ;

%%% Take the best C
[eMin,kMin] = min(min(eVal2))
[eMin,jMin] = min(min(eVal2'))

% Classify the test set using svmclassify
dTest   = svmclassify(svmStruct2{jMin,kMin},xTest);

%%% Error rates
eTest   = mean(dTest~=yTest);

%%% Display results
disp('Optimum hyperparameters:')
sigma_opt = sigma(jMin)
Copt = C(kMin)


% % Use a linear support vector machine classifier
% svmStruct = svmtrain(data(train,:),groups(train),'showplot',true);
% 
% % Add a title to the plot
% title(sprintf('Kernel Function: %s',...
%     func2str(svmStruct.KernelFunction)),...
%     'interpreter','none');
% 
% % Classify the test set using svmclassify
% classes = svmclassify(svmStruct,data(test,:),'showplot',true);
% 
% % See how well the classifier performed
% classperf(cp,classes,test);
% cp.CorrectRate

                     
                     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = polyprod(x,d);

%%% function y = polyprod(x,d); extends two-column matrix x with polynomial
%%% terms up to degree d.
y = x;

for i=2:d;
    
    x1e = repmat(x(:,1),1,i);
    y = [y x1e.*y(:,end-i+1:end) x(:,2).^i];

end






