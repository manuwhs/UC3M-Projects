

y1 = [-2 1 5 2 -1 4 7 7 7 7 5 4 3]; % Values of signal 1
n1 = [3 4 5 6 7 8 9 10 11 12 13 14 15];  % Domain of signal 1

y2 = [2 3 3 1];   % Values of signal 2
n2 = [8 9 10 11]; % Domain of signal 2

[ys, ns] = suma(y1,n1,y2,n2);  % Sum of signals
[yd, nd] = desplaza(y1,n1,4);  % Desplazamiento
[yi, ni] = interpola(y2,n2,3);  % Interpolado
[ydi, ndi] = diezma(y1,n1,3);  % Diezmado

figure();  % Create the main window
lim_x = [-3 16];  % Limites de graphicación
lim_y = [-3 11];

subplot(2,2,1); % 
stem(n1,y1);
xlim(lim_x)
ylim(lim_y)
title('y1[n]')

subplot(2,2,2)
stem(n2,y2);
xlim(lim_x)
ylim(lim_y)
title('y2[n]')

subplot(2,2,3)
stem(ns,ys);
xlim(lim_x)
ylim(lim_y)
title('ys = y1[n] + y2[n]')

subplot(2,2,4)
stem(nd,yd);
xlim(lim_x)
ylim(lim_y)
title('y1[n + 4]')



