x = [0:0.1:1 ones(1,5)]; % construimos la secuencia x
nx = 8:23;              % eje temporal para x
figure()          % la representamos graficamente
subplot(1,2,1)
stem(nx,x);
axis([-20 20 -0.2 1.2]);
title('x[n]')

y = x(end:-1:1); % abatimos los valores de la sennal
ny = -nx(end:-1:1); % tambien hay que abatir el vector de tiempos!!

subplot(1,2,2)
stem(ny,y);
axis([-20 20 -0.2 1.2]);
title('y[n]=x[-n]');  % titulo
