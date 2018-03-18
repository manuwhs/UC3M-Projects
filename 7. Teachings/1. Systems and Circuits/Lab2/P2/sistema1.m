function [ yo, no] = sistema1( x,xn )
    %% Primer sumando
    y1 = 0.2 * x.*x;
    n1 = xn;
    
    %% Segundo sumando
    y2 = - 0.3 * x.*x;
    n2 = xn;
    [y2, n2] = desplaza(y2, n2,-1);
    
    %% Tercer Sumando
    y3 = sin(xn/10*pi);
    n3 = xn;
    
    %% Cuarto Sumando
    y4 = - 0.4 * x;
    n4 = xn;
    [y4, n4] = desplaza(y4, n4,-2);
    
    %% Lo sumamos todo
    [yo, no] = suma(y1,n1,y2,n2);
    [yo, no] = suma(yo,no,y3,n3);
    [yo, no] = suma(yo,no,y4,n4);
end

