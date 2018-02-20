function [ fi ] = get_fi_matrix( I,N,T,A,B, alpha,beta,data )
    fi = zeros(I,I,N,T-1);
    for t = 1:T-1
        for n = 1:N
            for i = 1:I
                for j = 1:I
                    fi(i,j,n,t) = alpha(i,n,t) * A(i,j) * Prob_i_of_Y( B(j,:), data{n}(:,t+1)) * beta(j,n,t+1);
                end
            end
        end
    end
    
    for t = 1:T-1
        for n = 1:N
            % Normalize to get the actual fi
            fi(:,:,n,t) = fi(:,:,n,t)./sum(sum(fi(:,:,n,t)));  
        end
    end

    
end

