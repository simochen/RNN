function y = sigm(x)
%SIGM Compute sigmoid function
% y = SIGM(x) computes the sigmoid of x.

y = 1.0 ./ (1.0 + exp(-x));
end
