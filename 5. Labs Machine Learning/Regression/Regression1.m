% Part 1: Linear regression (weight space view)

randn('seed',0);
rand('seed',0);

%%%%%%%%%%%%%%%%%%%% INPUT DATA %%%%%%%%%%%%%%%%     1.a)

% Latent function f(X)
X = [];
X(:,2) = 0:0.1:2;   % X = [1, x1] 

n_xsamples = length(X(:,2));
X(:,1) = ones (n_xsamples,1);    % f(X)

%Noise
sigma_n = 0.2;  % Variance of the white noise.
m_n = 0;
noise = sigma_n*randn(n_xsamples,1) + m_n;

%Prior of weight vector
sigma_p = 2;    % Prior distribution of weight vector W = (w0 + X1*w1)
m_p = 0;        % Same distribution for all wights,  N(0,2)
cov_p = sigma_p*eye(2); % Weights are Independent

true_w = sigma_p*randn(2,1)+ m_p; % Realization of the prior weights
true_w = [1.6475; 0.8865];  % So that it fits the Exercises pdf example
% Resulting training input y = f(x) + n
y = X*true_w + noise;
plot(X(:,2), y);

%%%%%%%%%%%%%%%%%%%%%%%% p(w|X,y) %%%%%%%%%%%%%%%%%  1.b)
% Get the posterior model of the weights W

cov_p_inv = geninv(cov_p);             % Auxiliary matrices
A = (sigma_n^-1)*(X.'*X) + cov_p_inv;
A_inv = geninv (A); 

m_w = (sigma_n^-1)*A_inv*X.'*y;   % E of p(w|X,y)
cov_w = A_inv;                    % Var of p(w|X,y)

hold on
plot(X(:,2), X*m_w);

%%%%%%%%%%%%%% Confidence values %%%%%%%%%%%5%%   1.c) 
figure();
% The confidence values at 95% are obtaines as m +- 2*sigma
range_w1 = [m_w(1) - 2*sqrt(cov_w(1,1)), m_w(1) + 2*sqrt(cov_w(1,1))];
range_w2 = [m_w(2) - 2*sqrt(cov_w(2,2)), m_w(2) + 2*sqrt(cov_w(2,2))];
% Plotting the ranges
hold on
plot(X(:,2), X*[range_w1(1); range_w2(1)]);
hold on
plot(X(:,2), X*[range_w1(2); range_w2(2)]);
hold on;
plot(X(:,2), y,'x');
%%%%%%%%%%%% Sampling the posterior of W and perform regresion %%%%%%%%%%%   1.d) 
n_ws = 50;                  % Number of samples of the posterior
x_values = -1:0.1:3;        % X values used to estimate f(x)
n_test = length(x_values);

X_t = ones (n_test,2);    % f(X)
X_t(:,2) = x_values;

chol_cov_w = chol(cov_w); % Cholesky decomposition

% Rempat  copies the m_w.', n_ws times.
ws = repmat(m_w.',n_ws,1) + randn(n_ws,2)*chol_cov_w;
% Set of the ws values (samples of the posterior of w)
ws = ws.';

figure();
for i =1:n_ws
    y_ws = X_t*ws(:,i);
    hold on;
    plot(X_t(:,2),y_ws);
end
    
%%%% Sampling from the posterior of the function p(f|x,X_*,y) %%% 
% Where the X_* are the points -1:0.1:3 (New data to perform regression)

m_f = (sigma_n^-1)*A_inv*X.'*y;  % E of p(f|x,X,y)
m_f = X_t*m_f;

cov_f = X_t*A_inv*X_t.';        % Var of p(f|x,X,y)

figure();

plot (X_t(:,2), m_f);
hold on;

% Plot upper and lower variance bounds
[Xtest_sorted, order] = sort(X_t(:,2));

upper = zeros ( n_test, 1);
for i = 1:n_test
    upper(i) = m_f(i) + 2 * sqrt( cov_f(i,i));
end

upper_sorted = upper(order);
plot(Xtest_sorted,upper_sorted);
hold on;

lower = zeros ( n_test, 1);
for i = 1:n_test
    lower(i) = m_f(i) - 2 * sqrt(cov_f(i,i));
end

lower_sorted = lower(order);
plot(Xtest_sorted,lower_sorted);
hold on;

plot(Xtest_sorted,lower_sorted);
hold on;
plot(X(:,2), y,'x');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Using airline data %%%%%%%%%%%%%%%%%%%        1.f)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and normalize the data 
load('DatosLabReg.mat');
[N, D] = size(Xtrain); 
D = D + 1; 
Ntest = size(Xtest,1);
mx = mean(Xtrain,1); 
stdx = std(Xtrain,1,1);
X = [ones(N,1) (Xtrain - ones(N,1)*mx)./(ones(N,1)*stdx)];
Xtest = [ones(Ntest,1) (Xtest - ones(Ntest,1)*mx)./(ones(Ntest,1)*stdx)];
y = Ytrain;

%%%%%%%%%%%%%%%%%%%%% Get the prior of W and noise   %%%%%%%%%%%%%%%%% 1.g)

sigma_p = mean((X\y).^2); % Gross estimation of s02. Prior variance of W
sigma_n = 2*mean((y-X*(X\y)).^2); % Gross estimation of sn2. Prior noise variance.

cov_p = sigma_p*eye(D);  % Prior covariance of weight vector.

%%%%%%%%%%%%%%%%% Posterior of weight vector p(w|X,y) %%%%%%%%%%%%%%%%%%% 

cov_p_inv = geninv(cov_p);
A = (sigma_n^-1)*(X.'*X) + cov_p_inv;
A_inv = geninv (A); 

m_w = (sigma_n^-1)*A_inv*X.'*y;  % E of p(w|X,y)
m_w = m_w.';
cov_w = A_inv;                   % Var of p(w|X,y)

%%% Confidence values %%%  
% The confidence values at 95% are obtaines as m +- 2*sigma
range_w = zeros(D,2);
for i = 1:D
    range_w(i,:) = [m_w(i) - 2*sqrt(cov_w(i,i)), m_w(i) + 2*sqrt(cov_w(i,i))];
end

%%%%%%%%%%%%%%% Make the predictions %%%%%%%%%%%%%%%%%%%              1.h)

m_f = (sigma_n^-1)*A_inv*X.'*y;                 % E of p(f|x,X,y)
m_f = Xtest*m_f;

cov_f = Xtest*A_inv*Xtest.';        % Var of p(f|x,X,y)

m_y = m_f;
v_y = zeros (length(m_y),1);

for i=1:length(m_y)
    v_y(i) = (sigma_n) + cov_f(i,i); % Var(Y) = Var(f(X)) + Var(n)
end

MSE = mean((Ytest - m_y).^2);
NLPD = 0.5*mean((Ytest -m_y).^2./v_y+0.5*log(2*pi*v_y));

