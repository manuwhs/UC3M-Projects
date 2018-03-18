
function [N_errors] = get_Nerrors (Y_Real, Y_Pred)

[n1,m1] = size(Y_Real);
[n2,m2] = size(Y_Pred);

if (n1 ~= n2)
 %   exit(0);
end

N_errors = 0;
for i = 1:n1
    if (Y_Real(i) ~=  Y_Pred(i))
        N_errors = N_errors + 1;
    end
end

end
