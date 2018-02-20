
function [P1] = get_bin_prior (Y)

% It gives out the pripor probability of getting a 1. 

[N] = length(Y);
N_1s = 0;
for i = 1:N
    if (Y(i) == 1) 
      	N_1s = N_1s + 1;
    end
end

P1 = N_1s/N;