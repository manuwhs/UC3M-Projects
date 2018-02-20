function [] = plot_2D_bindata( X, Y )
%Plots the 2D datapoints X(i) with an "X" if Y(i) = 1 and with a "O" if
%Y(i) = 0
[n1,n_param1] = size(X);
[n2,n_param2] = size(Y);
if (n1 ~= n2)
 %   exit(0);
end

% Plotting the result
%figure();

for i = 1:n1
    hold on;
    if (Y(i) == 1) 
        scatter(X(i,1),X(i,2),'X','r')
    end
     if (Y(i) == 0) 
        scatter(X(i,1),X(i,2),'O','b')
     end
end

end
