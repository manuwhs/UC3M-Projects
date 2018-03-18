%% Sine things

N = 10000;   % Number of samples
P1 = 2;      % Number of periods s1

n = 0:N;
s1 = sin((2*pi*P1/N)*n);

figure();  % Create the main window

n = n*(2*pi*P1/N);

subplot(1,2,1); % 
plot(n,s1);
ylabel('y(t)')
xlabel('t (sec)')
title('Signal continua y(t) = sin(w x t)')

%% Sine things

N = 30;   % Number of samples
P1 = 2;      % Number of periods s1

n = 1:N;
s1 = sin((2*pi*P1/N)*n);


subplot(1,2,2)
stem(n,s1);
ylabel('y[n]')
xlabel('n')
title('Signal discreta y[n] = sin(w x [n x ts])' )



