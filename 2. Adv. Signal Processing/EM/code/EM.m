

% Input
% K = Cluster Number
% data = Data
% alpha = minimu step for convergence
% T = Max iterations
% 
% Output
% pimix = pi parameters
% theta =  theta parameters
% logL = complete log-likelihood of each iteration

function [pimix,theta,logl] = EM(K,data,alpha,T)

N = size(data,1); % Number of IDD samples
D = size(data,2); % Dimension of multidimensial bernoulli

%theta:  Matrix whose k-th column is the D-dimensional theta vector of
%parameters for the k-th component theta(:,k) = [theta_k1,..., theta_kD]

%**************************************
%*********** INITIALIZATION ***********
%**************************************

% Here we will initializ the  parameters of the mixture model, that is,the
% theta vectors of the K components and mixing coefficients %

% MIXING COEFICIENTS
% We set the mixing coefficients with uniform discrete distribution, this
% way, the a priori probability of any vector to belong to any component is
% the same.

pimix = ones(1,K);
pimix = pimix.*(1/K);

% THETA PARAMETERS OF THE K COMPONENTS
% Give random values to the theta parameters. Since in this case, all
% theta parameters theta(d,k), are the Expected value of a Bernoulli, we
% asign values to them at random accoding to a uniform continuous
% distribution in the support (0,1).

theta = rand(D,K);

%Initialize log-likelihood to 0
ll = 0;

%********************************************
%*********** ITERATIONS OF THE EM ***********
%********************************************

for t = 1:T             % For every iteration of the EM algorithm
                % disp(strcat('  # Iter: ',num2str(t)));
%******************************  
%*********** E Step ***********
%******************************
    % In this step we calculate the responsibility of each sample i to each
    % component k. This gives a measure of how likeky is sample i, to
    % belong to the component K.
    
    r = zeros(N,K);
    for i = 1:N
        Marginal_xi_probability = (sum(pimix(:).*(prod(theta(:,:).^repmat((data(i,:)==1)',1,K).*(1-theta(:,:)).^repmat((data(i,:)==0)',1,K)))'));
        for k = 1:K
            k_component_pdf = prod(theta(:,k).^((data(i,:)==1)').*(1-theta(:,k)).^((data(i,:)==0)'));
            r(i,k) = (pimix(k)*k_component_pdf)/Marginal_xi_probability;      
        end
    end
%*****************************   
%*********** M Step***********
%*****************************

    % In this step we calculate the next parameters of the mixture mdoel
    %Calculate new pimix and update
    for k = 1:K
        pimix(k) = sum(r(:,k))/N;
    end
               % disp('  pimix:');
               % disp(pimix);

    %Calculate new thetas and update
    for j = 1:D
        for k = 1:K
            theta(j,k) = sum(r(:,k).*data(:,j))/sum(r(:,k));
        end
    end
            % disp('  theta:');
            % disp(theta);

%*********************************** 
%****** Convergence Checking *******
%***********************************
    %Calculate log-likelihood
    new_ll = 0;
    for i = 1:N
        aux = 0;
        for k = 1:K
            aux = aux + (pimix(k)*prod(theta(:,k).^((data(i,:)==1)').*(1-theta(:,k)).^((data(i,:)==0)')));
        end
        new_ll = new_ll+log2(aux);
    end
            % disp('  new_ll:');
            % disp(new_ll);
    logl(t) = new_ll;

    if(abs(new_ll-ll) <= alpha)
        break;
    else
        ll = new_ll;
    end
    
end
             %  disp('  R:');
             %  disp(r);
end


