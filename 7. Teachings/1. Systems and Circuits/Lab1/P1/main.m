
close('all')

y1 = [-2 1 5]; % Values of signal 1
n1 = [3 4 5];  % Domain of signal 1

y2 = [2 3 3 1];   % Values of signal 2
n2 = [8 9 10 11]; % Domain of signal 2

[ys, ns] = suma(y1,n1,y2,n2);  % Sum of signals

figure();  % Create the main window
subplot(2,2,1); % 
stem(n1,y1);
xlim([-3 15])
ylim([-3 6])
title('y1[n]')

subplot(2,2,2)
stem(n2,y2);
xlim([-3 15])
ylim([-3 6])
title('y2[n]')

subplot(2,2,3)
stem(ns,xs);
xlim([-3 15])
ylim([-3 6])
title('ys = y1[n] + y2[n]')

[xd, nd] = desp(ys,ns,4);
subplot(2,2,4)
stem(nd,xd);
xlim([-3 15])
ylim([-3 6])
title('ys[n + 4]')


%% Sine things

N = 10000;   % Number of samples
P1 = 4;      % Number of periods s1
P2 = 2;      % Number of periods s2

n = 1:N;
s1 = sin((2*pi*P1/N)*n);
s2 = sin((2*pi*P2/N)*n);

y1 = s1 + s2;   % Sum of signals 
y2 = s1 .* s2;  % Product

figure();  % Create the main window

subplot(2,2,1); % 
plot(n,s1);
title('1st signal')

subplot(2,2,2)
plot(n,s2);
title('2nd signal')

subplot(2,2,3)
plot(n,y1);
title('Suma')

subplot(2,2,4)
plot(n,y2);
title('Producto')


%% Audio things

starwars = wavread('./Audios/starwars.wav');  % audioread si no funciona.
Lsw = length(starwars);   % Numero de muestras de starwars
n = 1:Lsw;               % Creamos el dominio discreto de la señal

ones_sig = ones(1,Lsw);
P = 5;
mask = ones_sig*0.5 + 0.5*cos((2*pi*P/Lsw)*n);
mask = mask.';

figure();  % Create the main window

subplot(2,2,1); % 
plot(n,starwars);
title('1st signal')

subplot(2,2,2)
plot(n,mask);
title('2nd signal')

subplot(2,2,3)
plot(n,starwars.*mask);
title('Product')


% starwars = starwars.*mask;
% escucha(starwars);

fil = hamming(100);  % Low pass filter
filtered_sw = conv(starwars, fil);
escucha(filtered_sw);

x = [0:0.1:1 ones(1,5)]; % construimos la secuencia x
nx = 1:16;              % eje temporal para x
figure()          % la representamos graficamente
subplot(1,2,1)
stem(nx,x);
axis([-1 20 -0.2 1.2]);
title('x[n]')

Ny = length(x)*2;
y = [];
for k=1:Ny   % interpolamos con ceros
   if rem(k,2) == 0
      y = [y x(k/2)];
   else
      y = [y 0];
   end
end
ny = 1:Ny;  % eje temporal para y

subplot(1,2,2) % representamos graficamente
stem(ny,y);
axis([-1 40 -0.2 1.2]);
title('y[n]=x[n/2]');  % etiqueta eje vertical
