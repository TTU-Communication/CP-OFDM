function [companding_sig] = companding(sig, mu)

    if isreal(sig)
        inSig = sig;
    else
        inSig = abs(sig);
    end

    peak = max(inSig);
    cSig = sign(inSig) .* peak .* log(1 + mu ./ peak .* abs(inSig)) ./ log(1 + mu);

    if isreal(sig)
        companding_sig = cSig;
    else
        companding_sig = cSig .* exp(1j .* angle(sig));
    end
    
end
