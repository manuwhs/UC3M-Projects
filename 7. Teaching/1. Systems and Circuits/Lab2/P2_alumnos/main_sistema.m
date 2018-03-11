
x1 = [-1 2 3 3.5 1 2.5];   % Values of signal 1
x1n = [-2 -1 0 1 2 3]; % Domain of signal 1

h = [2 -1 1];
hn = [2 3 4];

%% System function

[y1, y1n] = sistema1(x1, x1n);

figure();  % Create the main window
lim_x = [-3 14];  % Limites de graphicación
lim_y = [-6 6];

subplot(1,2,1); % 
stem(x1n,x1);
xlim(lim_x)
ylim(lim_y)
title('x1[n]')

subplot(1,2,2)
stem(y1n,y1);
xlim(lim_x)
ylim(lim_y)
title('y1[n]')

