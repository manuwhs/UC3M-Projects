load('marker.mat')
load('dna_amp_chr_17.mat')

N_trials = 2; %Number os times we run the EM to get the best
N_it_max = 15;  %Maximum number of iterations inside the EM.
N_K_max = 4;    % Number of maximum components

best_logll = [];    % Vector with loglikeihoods of the best run of the EM
best_theta = [];    % Matriz with thetas of the best run of the EM
best_pimix = [];    % Vector with the pimix the best run of the EM

best_final_logll = 0;  %Best final loglikelihood of any iteration

figure();                
                
for n_K = 2:N_K_max  % For each number of K components 
    % We make a first run of the EM
    [pimix,theta,logl] = EM(n_K,marker,-1, N_it_max); % dna_amp_chr_17   marker
    best_logll = logl;   
    best_theta = theta;    
    best_pimix = pimix;     
    best_final_logll = best_logll(N_it_max);

                 %   disp('  Best Logll:');
                 %   disp(best_final_logll);
    for i=1:N_trials
        [pimix,theta,logl] = EM(n_K,marker ,-1, N_it_max);  % dna_amp_chr_17   marker
        if (logl(N_it_max) > best_final_logll)
            best_logll = logl;   
            best_theta = theta;    
            best_pimix = pimix ;  
            best_final_logll = best_logll(N_it_max);
                  %  disp('  New best:');
                  %  disp(best_final_logll);
        end
    end
          disp('Parameters for K = '); disp(n_K);
          disp('theta' ); disp(best_theta);
          disp('pimix' );disp(pimix);
    plot (best_logll,'LineWidth',3,'Color',((n_K-1)/N_K_max)*[1 1 1]);
    hold on;
end


title('Best Complete Log Likelihoods in terms of K ')
xlabel('Iteration') % x-axis label
ylabel('Complete Log Likelihood') % y-axis label
legend('K = 2','K = 3','K = 4','Location','northwest');




