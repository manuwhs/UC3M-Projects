%%%%%% Using airline data %%%% 1.f)
randn('seed',0);

% Load and normalize the data 
load('DatosLabReg.mat');
Xtrain = Xtrain(:,1);
Xtest = Xtest(:,1);

[Ntest,D] = size(Xtest);
[Ntrain,N] = size(Xtrain); % D and N is the number of vector X.

%%%%%%% Normalization of the input data %%%%%%%%%%%%%

mx = mean(Xtrain,1);    % Mean of the input X values
stdx = std(Xtrain,1,1); % Standar deviation of the input X values
X = (Xtrain - ones(Ntrain,1)*mx)./(ones(Ntrain,1)*stdx);
Xtest = (Xtest - ones(Ntest,1)*mx)./(ones(Ntest,1)*stdx);
y = Ytrain;

%%%%%%%%%%%%% Parameters of the model %%%%%%%%%%%%%
mf = mean (y);     % Mean of the output

s02 = 1.5*var(y,1); % Gross estimation of s02, ell, and sn2
sn2 = s02/10;
ell = 0.05;  % Tuned by myself to make the function less smoth

%%%%%%%%%%%%%%%%%%%%%%%%% Parameter tunning %%%%%%%%%%%%%%%%%%%%%%%%%

% evidence([s02, ell, sn2], X, y);
% obj = @(hyper) evidence(hyper, X, y);
% hyper = fminsearch(obj, [s02, ell, sn2],optimset('Display','iter'));
% 
% s02 = hyper(1); 
% ell = hyper(2); 
% sn2 = hyper(3);

%%%%%%%%%%%%%%%%%% Calculate K (x,x), K(x*,x*) and K(x,x*) %%%%%%%%%%%%%%%%
%  K =  | K_tr  K_trte |
%       | K_tetr K_te  |

K_tr = zeros(Ntrain,Ntrain);
for i=1:Ntrain
    for j=1:Ntrain
        K_tr(i,j) = k_b(s02,ell,X(i,:),X(j,:));
    end
end
K_tr = K_tr + sn2 * eye(Ntrain);

K_te = zeros(Ntest,Ntest);
for i=1:Ntest
    for j=1:Ntest
        K_te(i,j) = k_b(s02,ell,Xtest(i,:),Xtest(j,:));
    end
end

K_trte = zeros(Ntrain,Ntest);
for i=1:Ntrain
    for j=1:Ntest
        K_trte(i,j) = k_b(s02,ell,X(i,:),Xtest(j,:));
    end
end

%%%%%%%%%%% Regresion of f* %%%%%%%%%%%%%%%%%%
Aux = K_trte.'*geninv(K_tr);
m_f = Aux*(Ytrain - mf*ones(Ntrain,1));  % E of p(f|x,X,y)

cov_f = K_te - Aux*K_trte;               % Var of p(f|x,X,y)

m_y = m_f + mf*ones(Ntest,1);
v_y = zeros (length(m_y),1);
for i=1:length(m_y)
    v_y(i) = sn2 + cov_f(i,i); % Var(Y) = Var(f(X)) + Var(n);
end

MSE = mean((Ytest - m_y).^2);
NLPD = 0.5*mean((Ytest -m_y).^2./v_y+0.5*log(2*pi*v_y));

%%%%%%%%%%%% Plotting of the 50 realizations of the GP %%%%%%%%%%%% c)
figure();
ns = 50;
cov_f_chol = chol(cov_f + eye(Ntest)*10e-4)' ;

[Xtest_sorted, order] = sort(Xtest);

for i = 1:ns
    r = randn(Ntest,1);  % Random realizations of N(0,1) to get the random part 
    % and then correlate them with the correlation matrix.
    GPs = m_y + cov_f_chol * r;
    GPs_sorted = GPs(order);
    plot(Xtest_sorted, GPs_sorted);
    hold on;
end
figure();

%%%%%%%%%%%%%%% Variance of y %%%%%%%%%%%%%%%%%%%%
[Xtest_sorted, order] = sort(Xtest);

hold on;
upper = zeros ( Ntest, 1);
for i = 1:Ntest
    upper(i) = m_y(i) + 2 * sqrt(v_y(i));
end

upper_sorted = upper(order);
plot(Xtest_sorted,upper_sorted);
hold on;

lower = zeros ( Ntest, 1);
for i = 1:Ntest
    lower(i) = m_y(i) - 2 * sqrt(v_y(i));
end
lower_sorted = lower(order);
plot(Xtest_sorted,lower_sorted);
hold on;

%%%%%%%%%%%%%%% Variance of f %%%%%%%%%%%%%%%%%%%%

hold on;
upper = zeros ( Ntest, 1);
for i = 1:Ntest
    upper(i) = m_y(i) + 2 * sqrt(cov_f(i,i));
end

upper_sorted = upper(order);
plot(Xtest_sorted,upper_sorted);
hold on;

lower = zeros ( Ntest, 1);
for i = 1:Ntest
    lower(i) = m_y(i) - 2 * sqrt(cov_f(i,i));
end
lower_sorted = lower(order);
plot(Xtest_sorted,lower_sorted);
hold on;

plot(X,Ytrain,'.');


