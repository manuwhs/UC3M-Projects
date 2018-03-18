

% Input
% S = Number of States
% data = Data
% alpha = minimu step for convergence (If negative it does not check it)
% R = Max iterations
% 
% Output
% pi = pi parameters
% A =  A parameters
% B =  B parameters
% logL = complete log-likelihood of each iteration

%  [pi,A,B,alfa] = HMM(5,observed,0.1,1)
function [pi,A,B,logl] = HMM(I,data,delta,R)

N = size(data,1);    % Number of Realizations of the HMM
D = size(data{1},1); % Dimension of multidimensial bernoulli
T = size(data{1},2); % Number of samples of the HMM


%theta:  Matrix whose k-th column is the D-dimensional theta vector of
%parameters for the k-th component theta(:,k) = [theta_k1,..., theta_kD]

%**************************************
%*********** INITIALIZATION ***********
%**************************************

% Here we will initialize the  parameters of the HMM, that is,the initial
% probabilities of the state "pi", the transition probabilities "A" and the
% parameters of the probability functions "B"

% Initial probabilities
% We set the Initial probabilities with uniform discrete distribution, this
% way, the a priori probability of any vector to belong to any component is
% the same.

pi = ones(1,I);
pi = pi.*(1/I);

% Transition probabilities "A"
% We set the Transition probabilities with uniform discrete distribution, this
% way, the a priori probability of going from a state i to a state j is
% the same, no matter the j.

A = ones(I,I);   %A(i,j) = aij = P(st = j | st-1 = i)  sum(A(i,:)) = 1
for i = 1:I
    A(i,:) =  A(i,:)*(1/I);
end

% Parameters of the probability functions "B"
% Give random values to the transit parameters. Since in this case, all
% theta parameters theta(d,k), are the Expected value of a Bernoulli, we
% asign values to them at random accoding to a uniform continuous
% distribution in the support (0,1).

B = rand(I,D);  % I vectors of D parameters

%Initialize log-likelihood to 0
ll = -100000000000000000;

%********************************************
%*********** ITERATIONS OF THE HMM ***********
%********************************************

for r = 1:R             % For every iteration of the EM algorithm
                % disp(strcat('  # Iter: ',num2str(r)));
%******************************  
%*********** E Step ***********
%******************************
% In this step we calculate the alfas, betas, gammas and fis matrices
 
alpha = get_alfa_matrix(I,N,T, A,B,pi,data);
beta = get_beta_matrix( I,N,T, A,B,data);
gamma = get_gamma_matrix( I,N,T, alpha,beta );
fi = get_fi_matrix( I,N,T,A,B, alpha,beta,data );

%*****************************   
%*********** M Step***********
%*****************************

% In this step we calculate the next parameters of the HMM

% Calculate new initial probabilities
    N_gamma = sum(sum(gamma(:,:,1)));
    N_i_gamma = zeros(I,1);
    for i = 1:I
        N_i_gamma(i) = sum(gamma(i,:,1));
        pi = N_i_gamma/N_gamma;
    end
    
%Calculate transition probabilities A
    for i = 1:I
        % Calculate vector ai = [ai1 ai2 ... aiJ]  sum(ai) = 1
        E_i_fi = sum(sum(sum(fi(i,:,:,:))));
        for j = 1:I
            E_ij_fi = sum(sum(fi(i,j,:,:)));
            A(i,j) = E_ij_fi/E_i_fi;
        end
    end
    
%Calculate the paramters B

    for i = 1:I
        N_i_gamma = sum(sum(gamma(i,:,:)));
        for d = 1:D
            aux = 0;
            for n = 1:N
                for t = 1:T
                    aux = aux + data{n}(d,t)*gamma(i,n,t);
                end
            end
             B(i,d) = aux/N_i_gamma;
        end
    end
  
%********************************************************* 
%****** Calculate Incomplete log-likelihood  *************
%*********************************************************

% Remember that the Incomplete log-likelihood could decrease with
% the number of iterations at some stage since the EM algorith 
% maximizes the Complete log-likelihood (They are different)

    %Calculate Incomplete log-likelihood with the Forward Algorithm
    new_ll = 0;
   
    for n = 1:N  % For every HMM sequence
        ll_n = 0;
        for i = 1:I
            ll_n = ll_n + alpha(i,n,T);
        end
        new_ll = new_ll + log2(ll_n);
    end
    
            % disp('  new_ll:');
            % disp(new_ll);
            
    logl(r) = new_ll;
    
%*********************************** 
%****** Convergence Checking *******
%***********************************

    % If delta < 0 we dont check convergence
    if (delta > 0)
        if(new_ll - ll <= delta)
            break;
        else
            ll = new_ll;
        end
    end
    
end
    

end


