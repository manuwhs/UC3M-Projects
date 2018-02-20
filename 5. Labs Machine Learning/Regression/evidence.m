function NLML = evidence(hyper, X, y)
%EVIDENCE Summary of this function goes here
%   Detailed explanation goes here

s02 = hyper(1); 
ell = hyper(2); 
sn2 = hyper(3); 

N = size(X,1);
mf = mean (y);     % Mean of the output

K = zeros(N,N);
for i=1:N
    for j=i:N
        K(i,j) = k_b(s02,ell,X(i,:),X(j,:));
        K(j,i) = K(i,j);
    end
end
aux1 =(1/2)*(y - ones(N,1) * (mf));
aux2 = K + sn2 * eye(N);
aux2 = aux2 + 10e-3 * eye(N);  %%%%%% Adding noise for possitive definitness %%%%

aux3 = 0;              %log(abs(aux2));
K_chol = chol(aux2);
for i = 1: N
    aux3 = aux3 + log(abs(K_chol(i,i)));
end

aux4 = (N/2) * log (2*pi);
NLML = aux1'*geninv(aux2)*aux1 + aux3 + aux4;

end

