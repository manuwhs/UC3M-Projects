function [ yo, no] = Myconv( x,xn, h, hn )
    yo = h(1).*x;  % The output of the system to the first value of x[n]
    no = xn + hn(1); % The domain is retarded as much as the system hn
    if (length(h) > 1)  % If the signal has more than one value
        for i = 2:length(h)
            xaux = x.*h(i);
            naux = xn + hn(1);
            [xaux, naux] = desplaza(xaux, naux, -i +1);

            [yo, no] = suma(yo, no, xaux, naux);
        end
    end
end

