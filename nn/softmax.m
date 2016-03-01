function y = softmax(x)
% Softmax function
    
    ex = exp(x);
    y = ex / sum(ex);
end