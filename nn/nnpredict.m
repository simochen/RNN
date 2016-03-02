function labels = nnpredict(nn, x)
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    
    [dummy, i] = max(nn.b{end},[],2);
    labels = i;
end
