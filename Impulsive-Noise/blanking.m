function outSig = blanking(inSig, threshold)
    outSig = inSig;

    idxBlank = abs(outSig) > threshold;
    outSig(idxBlank) = 0;
end