load('observed.mat');
[pi,A,B,logl]= HMM(6,observed,0.1,50);

figure();
plot (logl,'LineWidth',3);
title('Incomplete Log Likelihood ')
xlabel('Iteration') % x-axis label
ylabel('Incomplete Log Likelihood') % y-axis label

[S_MAP_SbS,S_ML_Vit,S_MAP_Vit] = Decoding(observed, A, B, pi);






