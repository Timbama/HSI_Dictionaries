function X_hat = morph(X)
%MORPH Summary of this function goes here
%   Detailed explanation goes here
r =2;
SE = strel('square',r);
%SE2 = strel('line',r,0);
X_hat = imclose(X,SE);
%X_hat = imerode(X_hat,SE2);    
end

