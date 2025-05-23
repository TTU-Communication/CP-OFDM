function [mapped_sig] = mapping_subcarrier(sig, FFT_size)
    [N, syms] = size(sig);
    if N == FFT_size
        mapped_sig = sig;
        return;
    end

    mapped_sig = zeros(FFT_size, syms);
    mapped_sig(2:(N / 2 + 1), :) = sig(1:(N / 2), :);
    mapped_sig((end - N / 2 + 1):end, :) = sig((N / 2 + 1):end, :);
end

