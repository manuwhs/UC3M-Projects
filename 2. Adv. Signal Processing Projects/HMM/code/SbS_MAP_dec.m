
function [S] = SbS_MAP_dec(Y,gamma,T)
% This function outputs the sequence S = {s1,..., sT} of the Step by Step MAP 
% probability of Y = {y1,..., yT}.
% For every observation yt, we have st = arg max (gamma(t))
% gamma(i,t)  Tells us the gamma at time t for the state i.
S = zeros (1,T);
for t = 1:T         % For every observation of the sequence
     S(t) =      st_FB_MAP_dec(t,gamma,T);
end