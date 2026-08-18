function outSig = clipping(inSig, threshold)
    outSig = inSig;

    idxClip = abs(outSig) > threshold;
    outSig(idxClip) = threshold .* exp(1j .* angle(outSig(idxClip)));
end