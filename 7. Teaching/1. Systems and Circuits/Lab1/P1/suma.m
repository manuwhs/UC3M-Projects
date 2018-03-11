function [ yo, no ] = suma(y1,n1,y2,n2)
    n_min = min(min(n1),min(n2));   % Get the limits of the domain
    n_max = max(max(n1),max(n2));
    no = [n_min:n_max];             % Create the domain
    
    % Add the proper number of 0s to the left and right of the first signal
    n0_iz = min(n1) - min(no);
    n0_de = max(no) - max(n1);
    y1 = [zeros(1,n0_iz), y1, zeros(1,n0_de)];
    
    % Add the proper number of 0s to the left and right of the second signal
    n0_iz = min(n2) - min(no);
    n0_de = max(no) - max(n2);
    y2 = [zeros(1,n0_iz), y2, zeros(1,n0_de)];
    
    yo = y1 + y2; % Add signal
    
end

