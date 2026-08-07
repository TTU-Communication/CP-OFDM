function outSig = cpRemover(inSig, cpLen)
    indices = repmat({':'}, 1, ndims(inSig));
    sigSampleLen = size(inSig, 1);
    indices{1} = (cpLen+1):sigSampleLen;
    outSig = inSig(indices{:});
end