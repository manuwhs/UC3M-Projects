
% [S_MAP_SbS,S_ML_Vit,S_MAP_Vit]= Decoding(observed, best_A, best_B, best_pi);

% This function makes 3 different decodifications:
%   Step By Step MAP decoder 
%   ML Viterbi decoder
%   MAP Viterbi decoder
function [S_MAP_SbS,S_ML_Vit,S_MAP_Vit] = Decoding(data, A, B, pi)

% Data values
N = size(data,1);    % Number of Realizations of the HMM
D = size(data{1},1); % Dimension of multidimensial bernoulli
T = size(data{1},2); % Number of samples of the HMM

%  HMM parameters
I = size(A,1);
alpha = get_alfa_matrix(I,N,T, A,B,pi,data);
beta = get_beta_matrix(I,N,T, A,B,data);
gamma = get_gamma_matrix( I,N,T, alpha,beta );

tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Step By Step MAP decoder %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S_MAP_SbS = zeros (N,T);

for n = 1:N   % For every sequence Y = {y1,..., yT}.
    Y_n = data{n}(:,:);
    S_MAP_SbS(n,:) = SbS_MAP_dec(Y_n,gamma(:,n,:),T);
end

toc

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% ML Viterbi decoder %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We compute it by means of the log-likelihood of Y given S
S_ML_Vit = zeros (N,T);

Past_Journeys = zeros(T,I);   % Over dimensioned to abarcar all cases
Journeys = zeros(T,I); 
Costs = zeros(T,I);  % We compute the costs in logarithmic form
Possible_journeys_c = zeros(1,I);

% Journeys(t,i) contains the survival path at time index t, that ends at
% state number i
% Costs(t,i) contains the cost of the survival path al time index t, 
% that ends atstate number i
% Possible_journeys_c(j) contains all the I possible journeys that go from
% Journey(t-1,:) to state(t) = i
% First step for t = 1

for n = 1:N
    for i = 1:I 
        Journeys(1,i) = i;
        Costs(1,i) = log2(Prob_i_of_Y( B(i,:), data{n}(:,1)));
    end
    Past_Journeys = Journeys;
    for t = 2:T           % For every time index
        for i = 1:I       % For every state(t) = i we want to get the survival journey for
             for j = 1:I  % For every Journey(t-1,j) we could come from
                Possible_journeys_c(j) = Costs(t-1,j) + log2(Prob_i_of_Y( B(i,:), data{n}(:,t)));
             end
             [Prob, best_j] = max (Possible_journeys_c(:));
             % Generate the new journey that ends in state i at time t as the
             % joining of the best previous survival journey and the state i
             Journeys(1:t-1,i) = Past_Journeys(1:t-1,best_j);
             Journeys(t,i) = i;
             Costs(t,i) = Costs(t-1,best_j) + Prob;
        end
        Past_Journeys = Journeys;
    end
   [Prob, best_S] = max (Costs(T,:));
    S_ML_Vit(n,:) = Journeys(:,best_S);    
end

toc

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% MAP Viterbi decoder %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S_MAP_Vit = zeros (N,T);

sigma = zeros(T,I); % Sigma matriz x of values
Survival_paths = zeros(T,I); % Sigma matriz x of values
Past_Survival_paths = zeros(T,I); % Sigma matriz x of values

aux_sigma_j = zeros(1,I); % Aux probabilities to get the best jouerney
for n = 1:N
    % Calculate sigma 1
    for i = 1:I 
        sigma(1,i) = Prob_i_of_Y( B(i,:), data{n}(:,1)) * pi(i);
        Survival_paths(1,i) = i;
    end
    
    Past_Survival_paths =  Survival_paths;
    % Calculate sigma t recursively with dynamic programming
    for t = 2:T   
        for i = 1:I         % For state i at time t 
            for j = 1:I     % We calculate the joint probability with the j posible journes at time t-1 
                aux_sigma_j(j) = A(j,i) * sigma(t-1,j);
            end
            [sigma_tj, best_j] = max (aux_sigma_j(:));
            % sti is the estimated state of the survival path
            sigma(t,i) = Prob_i_of_Y( B(i,:), data{n}(:,t)) * sigma_tj;
            Survival_paths(1:t-1,i) = Past_Survival_paths(1:t-1,best_j);
            Survival_paths(t,i) = i;
        end
        Past_Survival_paths = Survival_paths;
    end
    
    % The best Survival path is the one that maxizes the sigma(T,i)
    % It is the one that ends with the i, that is: Survival_paths(:,best_S)
    [best_sigma, best_S] = max (sigma(T,:));
    S_MAP_Vit(n,:) = Survival_paths(:,best_S);
    
end
toc

end
