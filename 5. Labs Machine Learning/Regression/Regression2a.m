%%%%%% Using airline data %%%% 1.f)

% Load and normalize the data 
load('DatosLabReg.mat');
[N, D] = size(Xtrain); 

Ntest = size(Xtest,1);
Ntrain = size(Xtrain,1);

mx = mean(Xtrain,1); 
stdx = std(Xtrain,1,1);
X = [ones(N,1) (Xtrain - ones(N,1)*mx)./(ones(N,1)*stdx)];
Xtest = [ones(Ntest,1) (Xtest - ones(Ntest,1)*mx)./(ones(Ntest,1)*stdx)];
y = Ytrain;

sigma_p = mean((X\y).^2); % Gross estimation of s02 and sn2
sigma_n = 2*mean((y-X*(X\y)).^2);

%%% Calculate K (x,x), K(x*,x*) and K(x,x*)

K_tr = zeros(Ntrain,Ntrain);
for i=1:Ntrain
    for j=1:Ntrain
        K_tr(i,j) = k_a(sigma_p,X(i,:),X(j,:));
    end
end
K_tr = K_tr + sigma_n * eye(Ntrain);

K_te = zeros(Ntest,Ntest);
for i=1:Ntest
    for j=1:Ntest
        K_te(i,j) = k_a(sigma_p,Xtest(i,:),Xtest(j,:));
    end
end

%%% Calculate K (x,x), K(x*,x*) and K(x,x*)
%  K =  | K_tr  K_trte |
%       | K_tetr K_te  |

K_trte = zeros( Ntrain,Ntest);

for i=1:Ntrain
    for j=1:Ntest
        K_trte(i,j) = k_a(sigma_p,X(i,:),Xtest(j,:));
    end
end

%%%%%%%%%%% Regresion of f* %%%%%%%%%%%%%%%%%%

Aux = K_trte.'*geninv(K_tr);
m_f = Aux*y;                    % E of p(f|x,X,y)

cov_f = K_te - Aux*K_trte;      % Var of p(f|x,X,y)

m_y = m_f;
v_y = zeros (length(m_y),1);

for i=1:length(m_y)
    v_y(i) = sigma_n + cov_f(i,i); % Var(Y) = Var(f(X)) + Var(n)
end

MSE = mean((Ytest - m_y).^2);
NLPD = 0.5*mean((Ytest -m_y).^2./v_y+0.5*log(2*pi*v_y));

