function [ gamma ] = get_gamma_matrix( I,N,T, alpha,beta )
    gamma = zeros(I,N,T);
    for t = 1:T
        for n = 1:N
            for i = 1:I
                gamma(i,n,t) = alpha(i,n,t) * beta(i,n,t);
            end           
        end
    end
    
    for t = 1:T
        for n = 1:N
            % Normalize to get the actual gamma
            gamma(:,n,t) = gamma(:,n,t)./sum(gamma(:,n,t));  
        end
    end

    
end

