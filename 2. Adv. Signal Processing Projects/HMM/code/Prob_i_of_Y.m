
% Probability of getting the vector y from a multinomial distribution with
% vector parameters bi

function [ Pi_y ] = Prob_i_of_Y( bi, y)
    bi = bi(:);  % We use the (:) so that we make sure they are column vectors
    y = y(:); 
    Pi_y = prod(bi.^((y == 1)).*(1-bi).^((y == 0)));
end
