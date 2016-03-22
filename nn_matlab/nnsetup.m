function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 0.5;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0;          %  Momentum
    nn.lambda                           = 0;            %  regularization
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    %nn.vecW                          = [];
    for i = 1 : nn.n-1   
        epsilon_init = 0.12;
        % weights and weight momentum
        nn.W{i} = rand(nn.size(i+1), nn.size(i)+1) * 2 * epsilon_init - epsilon_init;
        nn.vW{i} = zeros(size(nn.W{i}));  
        %nn.vecW = [nn.vecW ; nn.W{i}(:)];
    end
end
