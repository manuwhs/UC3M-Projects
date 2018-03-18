
%randn('seed',0);   % Uncomment if you want to always get the same realization

n = 100;  % Number of samples

ell = 20;   % Parameter ell 
sigma = 0.1;  % Autocorrelation paramter
sigma_n = 10e-5; % Noise 

GPs = zeros (n,1); % Vector with the values of the GP
m = zeros (n,1);    % Vector with the averages of the samples
% m = [-0.5:0.01:0.5].^2;    % Vector with the averages of the samples

K = zeros(n,n); % Correlation matrix

for i=1:n
    for j=1:n
          K(i,j) = k_b(sigma,ell,i,j);
    end
end

K = K + eye(n)*sigma_n; % Add noise so that it is positive definite matrix.
K_chol = chol(K).'; % Transpose so that it is a lower triangular matrix
%          |  v 0 0 0 0 |
%          |  v v 0 0 0 |
% K_chol = |  v v v 0 0 |
%          |  v v v v 0 |
%          |  v v v v v |

r = randn(n,1);  % Random realizations of N(0,1) to get the random part 
% and then correlate them with the correlation matrix.

for i = 1:n 
    GPs(i) = m(i) + K_chol(i,:) * r;
end

% This is the same as GPs = m + K_chol * r;
plot(GPs);






