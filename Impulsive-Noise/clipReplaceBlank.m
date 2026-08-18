function outSig = clipReplaceBlank(inSig, thresholdClip, thresholdReplace, thresholdBlank, sigPower)
    outSig = inSig;

    idxClip = abs(outSig) > thresholdClip;
    idxReplace = abs(outSig) > thresholdReplace;
    idxBlank = abs(outSig) > thresholdBlank;
    outSig(idxClip) = thresholdClip .* exp(1j .* angle(outSig(idxClip)));
    outSig(idxReplace) = (sqrt(pi .* sigPower ./ 4)) .* exp(1j .* angle(outSig(idxReplace)));
    outSig(idxBlank) = 0;
end