function U_soft = soft_comp(U, Tau)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
U_soft = wthresh(U,'s', Tau);
end

