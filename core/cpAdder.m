function outSig = cpAdder(inSig, cpLen)
    indices = repmat({':'}, 1, ndims(inSig));
    sigSampleLen = size(inSig, 1);
    indices{1} = (sigSampleLen-cpLen+1):sigSampleLen;
    outSig = cat(1, inSig(indices{:}), inSig);
end