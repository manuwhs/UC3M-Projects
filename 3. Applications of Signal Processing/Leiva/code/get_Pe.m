function [Error_Rate] = get_Pe(Y, T)

n_Sa = length(Y);

%%%%%%%%%% Calculate correct classification rate 
    N_missclass = 0;
    for i = 1 : n_Sa
        Actual_class = T(i);
        Predicted_class = sign(Y(i));
        if Actual_class~=Predicted_class
            N_missclass = N_missclass+1;
        end
    end
    Error_Rate = N_missclass/n_Sa;
 