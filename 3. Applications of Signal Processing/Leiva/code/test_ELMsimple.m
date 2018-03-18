load('abalone.mat')

Nh = 10;

ini = 5;
fin = 100;
Error_tst = zeros(1,fin + 1 - ini);
Error_tr = zeros(1,fin + 1 - ini);

for Nh = ini:fin
    [W,b,beta,Y] = simpleELM_train(X_tr, T_tr, Nh);
    Error_Rate_tr = get_Pe(Y,T_tr);
    Error_tr(Nh + 1 -ini) = Error_Rate_tr;
    
    [Y] = simpleELM_run(X_tst, W,b,beta);
    Error_Rate_tst = get_Pe(Y,T_tst);
    Error_tst(Nh + 1 -ini) = Error_Rate_tst;
    
end
plot(Error_tr,'b')
hold on
plot(Error_tst,'r')