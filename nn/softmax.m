function y = softmax(x)
%SOFTMAX Compute softmax function
% y = SOFTMAX(x) computes the softmax of matrix x.
    n = size(x,2);
    ex = exp(x);
    y = ex ./ (sum(ex,2)*ones(1,n));
end