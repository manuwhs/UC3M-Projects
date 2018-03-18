function [yo, no] = interpola(x,n, M)

    Ny = length(x)*M;
    yo = [];
    for k=1:Ny
        % interpolamos con ceros
        if rem((k-1),M) == 0    % El -1 de k es para que empiece en 0
            yo = [yo x((k-1)/M +1)];
        else
            yo = [yo 0];
        end
    end
    no = 1:Ny; % eje temporal para y
    no = no + n(1);  % Desplazamos el eje al comienzo de la señal x[n] de entrada
end
