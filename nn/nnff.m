function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
        % Calculate the unit's outputs (including the bias term)    
        nn.z{i} = nn.a{i - 1} * nn.W{i - 1}';      
        switch nn.activation_function 
            case 'sigm'
                nn.a{i} = sigm(nn.z{i});
            case 'tanh'
                nn.a{i} = tanh(nn.z{i});
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    
    nn.z{n} = nn.a{n - 1} * nn.W{n - 1}';
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.z{n});
        case 'softmax'
            nn.a{n} = softmax(nn.z{n});
    end

    %将y向量化
    for i = 1:nn.size(end)
        Y(:,i) = (y==i);
    end
    %error and loss
    nn.e = Y - nn.a{n};
    
    switch nn.output
        case 'sigm'
            nn.L = -sum(sum(Y.*log(nn.a{n})+(1-Y).*log(1-nn.a{n}))) / m; 
        case 'softmax'
            nn.L = -sum(sum(Y .* log(nn.a{n}))) / m;
    end
    if nn.lambda > 0
        for i = 1:n-1
            nn.L = nn.L + nn.lambda/(2*m)*sum(sum(nn.W{i}(:,2:end).^2));
        end
    end
end
