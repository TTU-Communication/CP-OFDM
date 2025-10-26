function [decompanding_sig] = decompanding(sig, mu)

    if isreal(sig)
        inSig = sig;
    else
        inSig = abs(sig);
    end

    peak = max(inSig);
    dSig = sign(inSig) .* peak ./ mu .* (exp(abs(inSig) .* log(1 + mu) ./ peak) - 1);

    if isreal(sig)
        decompanding_sig = dSig;
    else
        decompanding_sig = dSig .* exp(1j .* angle(sig));
    end
    
end
