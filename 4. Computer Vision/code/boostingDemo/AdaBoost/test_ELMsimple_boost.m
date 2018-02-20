load('abalone.mat')

Nh = 50;
Nrounds = 30;

classifier = AdaBoostELM_train(X_tr, T_tr,Nrounds, Nh);

[Y, Error_tr] = AdaBoostELM_eval(X_tr,T_tr, classifier);
Error_Rate_tr = get_Pe(Y,T_tr);

[Y, Error_tst] = AdaBoostELM_eval(X_tst,T_tst, classifier);
Error_Rate_tst = get_Pe(Y,T_tst);

plot(Error_tr,'b')
hold on
plot(Error_tst,'r')