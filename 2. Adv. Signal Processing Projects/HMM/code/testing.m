tic
load('observed.mat');

N_trials = 2;  %Number of times we run the BW to get the best redsult
N_it_max = 10;  %Maximum number of iterations inside the BW.
N_I_min = 2;  % Number of minimum states
N_I_max = 5;    % Number of maximum states

best_logll = [];    % Vector with loglikeihoods of the best run of the BW
best_pi = [];    % Matriz with pi of the best run of the BW
best_A = [];    % Matriz with A of the best run of the BW
best_B = [];    % Matriz with B of the best run of the BW

best_final_logll = 0;  %Best final loglikelihood of any iteration

figure();                
                
for n_I = N_I_min:N_I_max  % For each number of I states
    [pi,A,B,logl]=  run_several_HMM(n_I,observed,-1,N_it_max,N_trials) ;
    plot (logl,'LineWidth',3,'Color',((n_I-1)/N_I_max)*[1 1 1]);
    hold on;
end

title('Best Incomplete Log Likelihoods in terms of K ')
xlabel('Iteration') % x-axis label
ylabel('Incomplete Log Likelihood') % y-axis label

legend_strings = [];

for i = 1:N_I_max - N_I_min + 1 % For each number of S states
    legend_strings = [legend_strings; num2str(N_I_min + i -1, '%d')];
end
legend([legend_strings],'Location','northwest');

toc






