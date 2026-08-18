function outSig = clipBlank(inSig, thresholdClip, thresholdBlank)
    assert(thresholdClip <= thresholdBlank, ...
        'thresholdBlank must be >= thresholdClip.');

    tempSig = blanking(inSig, thresholdBlank);
    outSig = clipping(tempSig, thresholdClip);
end