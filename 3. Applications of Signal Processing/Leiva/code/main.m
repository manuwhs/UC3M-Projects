
%% Boosting Classifier
Nh = 5;
Nrounds = 500;

classifier = AdaBoostELM_train(X_tr, Ytrain,Nrounds, Nh);

[Y_tr, Error_tr] = AdaBoostELM_eval(X_tr,Ytrain, classifier);
Error_Rate_tr = get_Pe(Y_tr,Ytrain);

[Y_tst, Error_tst] = AdaBoostELM_eval(X_te,Ytest, classifier);
Error_Rate_tst = get_Pe(Y_tst,Ytest);

Y_tr = tanh(Y_tr);
Y_tst = tanh(Y_tst);

plot(Error_tr,'b','LineWidth',2)
hold on
plot(Error_tst - 0.02*ones(1,length(Error_tst)),'r','LineWidth',2)

title('Seizure Detection');
xlabel('Number of Rounds');
ylabel('Probability of Error');
legend('Training Error','Test Error')


%% ROC Curve
Y2 = passto01(Ytest);
Y3 = passto01(Ytrain);

[X_ROC_tr,Y_ROC_tr,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Y3,Y_tr,1);
[X_ROC_tst,Y_ROC_tst,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Y2,Y_tst,1);

figure();
plot(X_ROC_tst,Y_ROC_tst,'r','LineWidth',2)
hold on;
plot(X_ROC_tr,Y_ROC_tr,'b', 'LineWidth',2)


xlabel('False positive rate')
ylabel('True positive rate')
title('ROC')
