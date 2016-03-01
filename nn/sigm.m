function y = sigm(x)
%SIGMOID Compute sigmoid function
% y = SIGMOID(x) computes the sigmoid of x.

y = 1.0 ./ (1.0 + exp(-x));
end
