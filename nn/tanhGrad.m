function  y = tanhGrad(x)
%TANHGRAD returns the gradient of the tanh function
%evaluated at x

    y = 1 - tanh(x).^2;
end