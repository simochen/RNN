function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    l = nn.l;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.b{1} = x;

    %feedforward pass
    for i = 2 : n-1
        % Calculate the unit's outputs (including the bias term)    
        nn.a{i} = nn.b{i - 1} * nn.W{i - 1}';      
        switch nn.activation_function 
            case 'sigm'
                nn.b{i} = sigm(nn.a{i});
            case 'tanh'
                nn.b{i} = tanh(nn.a{i});
        end
        
        %Add the bias term
        nn.b{i} = [ones(m,1) nn.b{i}];
    end
    
    nn.a{n} = nn.b{n - 1} * nn.W{n - 1}';
    switch nn.output 
        case 'sigm'
            nn.b{n} = sigm(nn.a{n});
        case 'softmax'
            nn.b{n} = softmax(nn.a{n});
    end

    %error and loss
    nn.e = nn.b{n} - y;
    
    switch nn.output
        case 'sigm'
            nn.L = -sum(nn.b{n}.*log(y)+(1-nn.b{n}).*log(1-y)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.b{n}))) / m;
    end
end
