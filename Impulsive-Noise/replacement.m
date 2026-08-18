function outSig = replacement(inSig, threshold, sigPower)
    outSig = inSig;

    idxReplace = abs(outSig) > threshold;
    outSig(idxReplace) = (sqrt(pi .* sigPower ./ 4)) .* exp(1j .* angle(outSig(idxReplace)));
end