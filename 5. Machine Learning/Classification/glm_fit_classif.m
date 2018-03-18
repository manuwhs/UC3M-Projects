
function [Y] = glm_fit_classif (B,X)
% This function classifies the vector X using logistic regression
% According to ML probability given by sigmoid, it is the same a linear
% decision boundary with proyection on B.

[n1,n_param1] = size(X);
[n2,n_param2] = size(B);

if (n1 ~= n2)
 %   exit(0);
end

Y = zeros(1,n1);
for i = 1:n1
    XW = ([1, X(i,:)])*(B);
    if (XW <= 0) 
        Y(i) = 0;
    else
        Y(i) = 1;
    end
end

end

