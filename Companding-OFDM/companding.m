function [companding_sig] = companding(sig, mu)
    if isreal(sig)
        peak = max(sig);
        companding_sig = sign(sig) .* peak .* log(1 + mu ./ peak .* abs(sig)) ./ log(1 + mu);
    else
        sig_i = real(sig);
        sig_q = imag(sig);
        peak_i = max(sig_i);
        peak_q = max(sig_q);
        companding_sig_i = sign(sig_i) .* peak_i .* log(1 + mu ./ peak_i .* abs(sig_i)) ./ log(1 + mu);
        companding_sig_q = sign(sig_q) .* peak_q .* log(1 + mu ./ peak_q .* abs(sig_q)) ./ log(1 + mu);
        companding_sig = companding_sig_i + 1i .* companding_sig_q;
    end
    
end
