function [ beta ] = get_beta_matrix( I,N,T, A,B,data )
    beta = zeros(I,N,T);
    % Calculate  alfa(1,:,:)
    
    for n = 1:N
        for i = 1:I
            beta(i,n,T) = 1;
        end
    end
    
    % Calculate the rest of the betas recursively
    for t = T-1:-1:1
        for n = 1:N
            for i = 1:I
                for j = 1:I
                    beta(i,n,t) = beta(i,n,t)+ A(i,j)* Prob_i_of_Y( B(j,:), data{n}(:,t)) * beta(j,n,t+1);
                end
            end
        end
    end
    
end

