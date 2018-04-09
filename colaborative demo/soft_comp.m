function x = soft_comp(b,lambda)

% Set the threshold
th = lambda;
shape = size(b);
x = zeros(shape);
xdim = size(0);
y = size(1);
for i = 1:xdim
    for j = 1:y
      x(i,j) = sign(b(i,j))*max((abs(b(i,j))-th),0);
    end

end
end