function [noise] = noise_AWGN(sig_size, snr, sig_power, sig_sample)
    snr_linear = 10 .^ (snr / 10);
    noise_power = sig_power / snr_linear;
    noise = sqrt(noise_power) .* randn(sig_size, 'like', sig_sample);
end
