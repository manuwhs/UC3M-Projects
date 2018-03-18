
function [Y] = glm_fit_classif_MAP (B,X,Prior_1)
% This function classifies the vector X using logistic regression
% According to MAP probability (dont know if it even makes sense coz
% the probability digmoid function is invented. Not real actually.
[n1,n_param1] = size(X);
[n2,n_param2] = size(B);

if (n1 ~= n2)
 %   exit(0);
end

Y = zeros(1,n1);
for i = 1:n1
    MAP_1 = get_glmfit_MAP_1 (X(i,:),B,Prior_1);
    if (MAP_1 <= 0.5) 
        Y(i) = 0;
    else
        Y(i) = 1;
    end
end

end

