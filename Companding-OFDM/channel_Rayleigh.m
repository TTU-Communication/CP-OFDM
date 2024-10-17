function [out_sig, channel_response] = channel_Rayleigh(sig, channel_length, channel_power, useful_size)
    [sig_row, sig_column] = size(sig);
    multipath = sqrt(channel_power) * randn(channel_length, sig_column, 'like', sig(1));
    convLen = sig_row + channel_length - 1;
    through_channel_signal = ifft(fft(sig, convLen, 1) .* ...
                        fft(multipath, convLen, 1), convLen, 1);

    channel_response = fft(multipath, useful_size, 1);
    out_sig = through_channel_signal(1:sig_row, :);
end
