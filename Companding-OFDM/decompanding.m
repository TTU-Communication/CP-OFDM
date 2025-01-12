function [decompanding_sig] = decompanding(sig, mu)
    if isreal(sig)
        peak = max(sig);
        decompanding_sig = sign(sig) .* peak ./ mu .* (exp(abs(sig) .* log(1 + mu) ./ peak) - 1);
    else
        sig_i = real(sig);
        sig_q = imag(sig);
        peak_i = max(sig_i);
        peak_q = max(sig_q);
        decompanding_sig_i = sign(sig_i) .* peak_i ./ mu .* (exp(abs(sig_i) .* log(1 + mu) ./ peak_i) - 1);
        decompanding_sig_q = sign(sig_q) .* peak_q ./ mu .* (exp(abs(sig_q) .* log(1 + mu) ./ peak_q) - 1);
        decompanding_sig = decompanding_sig_i + 1i .* decompanding_sig_q;
    end
    
end
