function [Y0] = passto01(Y)

n_Sa = length(Y);
Y0 = zeros(1,n_Sa);
%%%%%%%%%% Calculate correct classification rate ç
for i = 1 : n_Sa
        if (Y(i) > 0)
            Y0(i) = 1;
        end
end
