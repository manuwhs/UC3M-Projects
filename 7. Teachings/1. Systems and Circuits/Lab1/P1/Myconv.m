function [ o ] = Myconv( x, h )
%MYCONV Summary of this function goes here
%   Detailed explanation goes here
    o = zeros(1,length(x)+ length(h)- 1); 
    for i = 1:length(x)
        o = o + [zeros(1,i-1), x(i).*h, zeros(1,length(x)-(i))];
    end
end

