
% Runs the HMM N times and chooses the realization with the least loglihood

function [best_pi,best_A,best_B,best_logl] = run_several_HMM(I,data,delta,R,N)
    
% We make a first run of the HMM
    [pi,A,B,logl]= HMM(I,data,delta,R);
    best_logl = logl;   
    best_pi = pi;    
    best_A = A;       
    best_B = B;     
    best_final_logll = logl(R);
    
if (N > 1)
    
    for i = 1:N - 1
        [pi,A,B,logl] = HMM(I,data,delta,R); 
        if (logl(R) > best_final_logll)
            best_logl = logl;   
            best_pi = pi;    
            best_A = A;       
            best_B = B;     
            best_final_logll = logl(R);
        end
    end

end

end


