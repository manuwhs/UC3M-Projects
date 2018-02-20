%http://www.tsc.uc3m.es/~jose/data_BCI.mat

n_bits = 72 ;
n_exp = 63;
n_signals = 58;
n_samples = 150;
n_letters = 36;
X_train = zeros (n_bits * n_exp ,n_signals * n_samples );

for exp_i = 1:n_exp
    for bit_i = 1:n_bits
        for signal_i = 1:n_signals  
            X_train(bit_i + (exp_i - 1)*n_bits,1 +(signal_i - 1)*n_samples:1 +(signal_i - 1)*n_samples + (n_samples-1)) = data(cue(bit_i,exp_i):cue(bit_i,exp_i)+(n_samples-1),signal_i,exp_i);
        end
    end
end

Y_train = zeros(1,n_exp*n_bits);
for i=1:n_exp
    for j=1:n_bits
    Y_train(j + (i-1)*n_bits) = isT(j,i);
    end
end

%% WE calculeta Xtest

X_test = zeros (n_bits * 1 ,n_signals * n_samples );

exp_i = 64;
for bit_i = 1:n_bits
    for signal_i = 1:n_signals  
        X_test(bit_i,1 +(signal_i - 1)*n_samples:1 +(signal_i - 1)*n_samples + (n_samples-1)) = data(cue(bit_i,exp_i):cue(bit_i,exp_i)+(n_samples-1),signal_i,exp_i);
    end
end



%id is a vector of size X_test which contains for every X_test vector, the
%X_train vector that is closest.  Its a vector of indexes of X_train

id = knnsearch (X_train, X_test);

Y_test = zeros(1,72);
for i=1:72
    Y_test(i) = Y_train(id(i));
end

Y_real_test = zeros(1,1*n_bits);
for i=1:1
    for j=1:n_bits
        Y_real_test(j + (i-1)*n_bits) = isT(j,i+63);
    end
end

N_errors = 0
for i=1:72
    if (Y_real_test(i) ~=  Y_test(i))
        N_errors = N_errors + 1
    end
end

% Build classsifier that predicts que letter with the least hamming
% distance

CB_exp = CB(:,:,64);
dist_v = zeros(1,n_letters)
%CB (36,72,64) contains:
% Every experiment has diffent 72 bit values for the letters but the
% letters are at the same positions.
% So this tells us for every experiment, the 72bit code of each of the 36
% words.

for i=1:n_letters 
    N_mismatch = 0  %hamming distance
    for j=1:72
        if (Y_test(j) ~= CB_exp(i,j))
            N_mismatch = N_mismatch + 1
        end
    end
    dist_v(i) = N_mismatch;
end
    
[A,I] = min(dist_v);

Y_character_classifies = I;


% NEXT CLASSIFIER
% training -> glmfit
% testing -> glmval

D_f = 10;
N_vectors = n_bits * n_exp;
N_v_D = floor(N_vectors/D_f);
X_train_D = zeros(N_v_D,n_signals * n_samples);
Y_train_D = zeros(N_v_D,1);

for i=1:N_v_D
    X_train_D(i,:) = X_train(1+D_f*i,:);
    Y_train_D(i) = Y_train(1+D_f*i);
end

B = glmfit(X_train_D,Y_train_D,'binomial'); % Calculates the weight vector values of the logistic regression

yClass = zeros (1,400);
B_points = zeros (1,400);

for i=1:400
   B_point = B(1) + B(2)*xTrain(i,1) + B(3)*xTrain(i,2);
   B_points(i) = B_point;
   if (B_point < 0) 
       yClass(i) = 0;
   end
   if (B_point >= 0) 
       yClass(i) = 1;
   end  
end


%  for i=1:400
%  vector_Class1(i)=
%  end



