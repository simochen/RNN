function y = sigmGrad(x)
%SIGMGRAD returns the gradient of the sigmoid function
%evaluated at x
%   y = SIGMGRAD(x) computes the gradient of the sigmoid function
%   evaluated at x. This should work regardless if x is a matrix or a
%   vector. In particular, if x is a vector or matrix, you should return
%   the gradient for each element.


y = sigm(x).*(1-sigm(x));


end
