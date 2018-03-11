function [ yo, no ]  = abate(x,n)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    yo = x(end:-1:1); % abatimos los valores de la sennal
    no = -n(end:-1:1); % tambien hay que abatir el vector de tiempos!!
end

