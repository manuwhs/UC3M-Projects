
function [st] = st_FB_MAP_dec(t,gamma,T)
% This function outputs the MAP state at time t, st, given all the
% observations as   st = arg max (gamma(t))
% gamma(i,t)  Tells us the gamma at time t for the state i.

for i = 1:T     % For every observation of the sequence
    [gmax, idx] = max(gamma(:,t));
    st =  idx;
end

end