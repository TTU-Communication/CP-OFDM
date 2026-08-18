function outSig = deepClip(inSig, threshold, mu)
    tempSig = blanking(inSig, (1 + mu) / mu * threshold);
    outSig = tempSig;

    idxDeepClip = abs(tempSig) > threshold;
    outSig(idxDeepClip) = (threshold - mu .* (abs(outSig(idxDeepClip)) - threshold)) ...
                            .* exp(1j .* angle(outSig(idxDeepClip)));
end