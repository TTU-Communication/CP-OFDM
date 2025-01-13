function [demapped_sig] = demapping_subcarrier(sig, N)
    [FFT_size, syms] = size(sig);
    if FFT_size == N
        demapped_sig = sig;
        return;
    end
    
    demapped_sig = zeros(N, syms);
    demapped_sig(1:(N / 2), :) = sig(2:(N / 2 + 1), :);
    demapped_sig((N / 2 + 1):end, :) = sig((end - N / 2 + 1):end, :);
end

