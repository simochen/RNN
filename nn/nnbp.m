function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    m = size(nn.b{1},1);
    d{n} = nn.e;
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = sigmGrad(nn.a{i});
            case 'tanh'
                d_act = tanhGrad(nn.a{i});
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = d{i + 1} * nn.W{i} .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = d{i + 1}(:,2:end) * nn.W{i} .* d_act;
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.b{i}) / m;
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.b{i}) / m;      
        end
    end
end
