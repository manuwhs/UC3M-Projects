function [yo, no] = diezma(x,n, M)
    %DIEZMA Summary of this function goes here
    %   Detailed explanation goes here
    Ny = floor(length(x)/M); % Cogemos una de cada M muestras
    yo = [];
    for k=1:Ny
        yo = [yo x((k-1)*M + 1)];    % Anadimos la nueva muestra  El -1 de k es para que empiece en 0
    end
    no = 1:Ny; % eje temporal para y
    no = no + n(1);  % Desplazamos el eje al comienzo de la señal x[n] de entrada

end

