function labels = nnpredict(nn, x)
    nn = nnff(nn, x, zeros(size(x,1), 1));
    k = size(nn.a{end},2);
    if k==1
        i = nn.a{end} > 0.5;
        i = 2.^(1-i);
    else
        [dummy, i] = max(nn.a{end},[],2);
    end
    labels = i;
end
