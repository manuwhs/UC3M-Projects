
x1 = [-1 2 3 3.5 1 2.5];   % Values of signal 1
x1n = [-2 -1 0 1 2 3]; % Domain of signal 1

h = [2 -1 1];
hn = [2 3 4];

%% Convolution using matlab functions
yc = conv(x1,h);
nc = 1:length(yc);
nc = nc + hn(1) + x1n(1) - 1;

% [yc, nc] = conv();

[yc2, nc2] = Myconv(x1, x1n, h, hn);

figure();  % Create the main window
lim_x = [-3 9];  % Limites de graphicación
lim_y = [-6 9];

subplot(2,2,1); % 
stem(x1n,x1);
xlim(lim_x)
ylim(lim_y)
title('x1[n]')

subplot(2,2,2)
stem(hn,h);
xlim(lim_x)
ylim(lim_y)
title('h[n]')

subplot(2,2,3)
stem(nc,yc);
xlim(lim_x)
ylim(lim_y)
title('yc = h[n] * x1[n]')

subplot(2,2,4)
stem(nc2,yc2);
xlim(lim_x)
ylim(lim_y)
title('yc2 = h[n] * x1[n]')


