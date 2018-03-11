
x1 = [-1 2 3 3.5 1 2.5];   % Values of signal 2
x1n = [-2 -1 0 1 2 3]; % Domain of signal 2

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
lim_y = [-3 4];


%% Convolution using matlab functions


figure();  % Create the main window
lim_x = [-3 9];  % Limites de graphicación
lim_y = [-6 9];

x = x1;
xn = x1n;
for i = 1:length(h)
    xaux = x.*h(i);
    naux = xn + hn(1);
    [xaux, naux] = desplaza(xaux, naux, -i +1);
    
    subplot(4,1,i); % 
    stem(naux,xaux, 'LineWidth',2);
    xlim(lim_x)
    ylim(lim_y)
    texto = strcat('x[n - ', num2str(i + 1));
    texto = strcat(texto,'] h[');
    texto = strcat(texto, num2str(i + 1));
    texto = strcat(texto, + ']');
    title(texto)
end

subplot(4,1,4); % 
stem(nc,yc,'r','LineWidth',3);
xlim(lim_x)
ylim(lim_y)

title('y[n]')


