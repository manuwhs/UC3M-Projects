function  [kf] = k_a( sigma2,x_i,x_j)
%K_A Summary of this function goes here
%   Covariance function for 2.a)
    kf = sigma2 *( x_i * x_j.');
end

