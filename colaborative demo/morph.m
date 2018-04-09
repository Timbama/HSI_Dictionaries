function X_hat = morph(X)
%MORPH Summary of this function goes here
%   Detailed explanation goes here
r =2;
SE = strel('square',r);
%X = X';
%SE2 = strel('line',r,0);
%I = M*X
X_hat = imopen(X,SE);
%X_hat = X_hat';
%X_hat = imerode(X_hat,SE2);    
end

