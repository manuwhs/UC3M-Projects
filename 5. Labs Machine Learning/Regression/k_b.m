function  [kf] = k_b( sigma2,ell,x_i,x_j)
%K_A Summary of this function goes here
%   Covariance function for 2.b)
    diff = (x_i - x_j)*(x_i - x_j)';
    kf = sigma2 * exp(-(diff/ell));
end

