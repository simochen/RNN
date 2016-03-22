function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    m = size(nn.a{1},1);
    d{n} = nn.e;
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = sigmGrad(nn.z{i});
            case 'tanh'
                d_act = tanhGrad(nn.z{i});
        end
        
        % Backpropagate first derivatives
        d{i} = (d{i + 1} * nn.W{i}(:,2:end)) .* d_act;

    end

    for i = 1 : (n - 1)
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / m;
        nn.dW{i} = nn.dW{i} + nn.lambda / m * [zeros(nn.size(i+1),1) nn.W{i}(:,2:end)];
    end
end
