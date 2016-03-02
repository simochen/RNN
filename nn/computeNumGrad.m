function numgrad = computeNumGrad(func, nn)
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

% Notes: The following code implements numerical gradient checking, and 
%        returns the numerical gradient.It sets numgrad(i) to (a numerical 
%        approximation of) the partial derivative of J with respect to the 
%        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
%        be the (approximately) the partial derivative of J with respect 
%        to theta(i).)
%       
nn1 = nn;
nn2 = nn;
%numerical gradient
numgrad = [];
eps = 1e-4;
for i = 1 : nn.n-1
    for k = 1 : nn.size(i)+1
        for j = 1 : nn.size(i+1)
            nn1.W{i}(j,k) = nn.W{i}(j,k) - eps;
            nn2.W{i}(j,k) = nn.W{i}(j,k) + eps;
            nn1 = func(nn1);
            nn2 = func(nn2);
            numgrad = [numgrad ; (nn2.L - nn1.L) / (2*eps)];
            nn1.W{i}(j,k) = nn.W{i}(j,k);
            nn2.W{i}(j,k) = nn.W{i}(j,k);
        end
    end
end

end
