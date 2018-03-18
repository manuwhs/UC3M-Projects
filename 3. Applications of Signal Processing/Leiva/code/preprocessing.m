%http://www.tsc.uc3m.es/~jose/data_BCI.mat
%First Assingment -> Based on the power of the 16 separate channels

load('SeizurePredData.mat')
n_tr = 800;
n_te = 480;

n_ch = 16;
n_samples = 400;
sample_fr = 400; %Hz 

X_tr_pow = zeros (n_tr,n_ch);
X_te_pow = zeros (n_te,n_ch);

% TRAINING DATA TREATMENT
% We calculate the energy of the channel = sum(N samples)
for tr_i = 1:n_tr
    for ch_i = 1:n_ch
        for sample_i = 1:n_samples 
           X_tr_pow(tr_i,ch_i) = X_tr_pow(tr_i,ch_i) + Xtrain(sample_i,ch_i,tr_i).^2;
        end
    end
end

% We calculate the power (which is gonna be the same as the Energy coz the
% signals are 1 second long.
for tr_i = 1:n_tr
    for ch_i = 1:n_ch
        X_tr_pow(tr_i,ch_i) = X_tr_pow(tr_i,ch_i)/(n_samples/sample_fr);
    end
end

% TESTING DATA TREATMENT
% We calculate the energy of the channel = sum(N samples)
for te_i = 1:n_te
    for ch_i = 1:n_ch
        for sample_i = 1:n_samples 
           X_te_pow(te_i,ch_i) = X_te_pow(te_i,ch_i) + Xtest(sample_i,ch_i,te_i).^2;
        end
    end
end

% We calculate the power (which is gonna be the same as the Energy coz the
% signals are 1 second long.
for te_i = 1:n_te
    for ch_i = 1:n_ch
        X_te_pow(te_i,ch_i) = X_te_pow(te_i,ch_i)/(n_samples/sample_fr);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CORRELATION AMONG CHANNELS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% R = corrcoef(X)
% This MATLAB function returns a matrix R of correlation coefficients calculated
% from an input matrix X whose rows are observations and whose columns are
% variables.

X_tr_corr = zeros (n_tr,(n_ch*(n_ch-1))/2);
X_te_corr = zeros (n_te,(n_ch*(n_ch-1))/2);

for tr_i = 1:n_tr
    aux = 1;
    Corr_matrix = corrcoef (Xtrain(:,:,tr_i));
    for ch_i = 1:n_ch-1
        for ch_j = ch_i:n_ch-1
            X_tr_corr(tr_i, aux) = Corr_matrix (ch_i,ch_j);
            aux = aux + 1;
        end
    end
end

for te_i = 1:n_te
    aux = 1;
    Cov_matrix = corrcoef (Xtest(:,:,te_i));
    for ch_i = 1:n_ch-1
        for ch_j = ch_i:n_ch-1
            X_te_corr(te_i, aux) = Corr_matrix (ch_i,ch_j);
            aux = aux + 1;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Covariance AMONG CHANNELS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for tr_i = 1:n_tr
    Cov_matrix = cov (Xtrain(:,:,tr_i));
    X_tr_eigen(tr_i, :) = eig(Cov_matrix);
end

for te_i = 1:n_te
    aux = 1;
    Cov_matrix = cov (Xtest(:,:,te_i));
    X_tr_eigen(te_i, :) = eig(Cov_matrix);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frecuency Channgels Energy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_tr_eigen = [];
for tr_i = 1:n_tr
    Cov_matrix = cov (Xtrain(:,:,tr_i));
    X_tr_eigen(tr_i, :) = eig(Cov_matrix);
end

X_te_eigen = [];

for te_i = 1:n_te
    Cov_matrix = cov (Xtest(:,:,te_i));
    X_te_eigen(te_i, :) = eig(Cov_matrix);
end

% X_tr = [X_tr_pow, X_tr_corr, X_tr_eigen];
% X_te = [X_te_pow, X_te_corr, X_te_eigen];
% X_tr = [X_tr_pow];
% X_te = [X_te_pow];

% X_tr = [X_tr_corr, X_tr_eigen];
% X_te = [X_te_corr, X_te_eigen];

X_tr = [X_tr_pow, X_tr_eigen];
X_te = [X_te_pow, X_te_eigen];

%% Normalize between -1 and 1
X_min = min(X_tr);
X_max = max(X_tr);

[n_samples, n_dim] = size(X_tr);

% There are comoponents with no variance 
for i = 1:n_dim
    
    if (X_min(i) ~= X_max(i))
        X_tr (:,i) = X_tr (:,i) - (X_min(i) + (X_max(i) - X_min(i))/2);
        X_tr (:,i) = X_tr (:,i) / ((X_max(i) - X_min(i))/2);

        X_te (:,i) = X_te (:,i) - (X_min(i) + (X_max(i) - X_min(i))/2);
        X_te (:,i) = X_te (:,i) / ((X_max(i) - X_min(i))/2);
    end
end


for i = 1:n_tr
    if(Ytrain(i) == 0)
        Ytrain(i) = -1;
    end
end

for i = 1:n_te
    if(Ytest(i) == 0)
        Ytest(i) = -1;
    end
end
Xtrain = X_tr
Xtest = X_te


save('data.mat','Xtrain','Xtest','Ytrain','Ytest' ); % function form


