function [ alfa ] = get_alfa_matrix( I,N,T, A,B,pi,data )
    alfa = zeros(I,N,T);
    % Calculate  alfa(1,:,:)
    
    for n = 1:N
        for i = 1:I
            alfa(i,n,1) = pi(i)*Prob_i_of_Y( B(i,:), data{n}(:,1));
        end
    end
    
    % Calculate the rest of the alfas recursively
    for t = 2:T
        for n = 1:N
            for i = 1:I
                for j = 1:I
                    alfa(i,n,t) = alfa(i,n,t) + A(j,i)*alfa(j,n,t-1);
                end
                alfa(i,n,t) = Prob_i_of_Y( B(i,:), data{n}(:,t)) * alfa(i,n,t);
            end
        end
    end
    
end

