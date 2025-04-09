function [nullSCIdx] = getNullSCIndex(fftSize)
    %GETNULLSCINDEX get the correspond null subcarrier index to fft
    %size
    %   Avaliable FFT size: 32, 64, 128, 256, 512, 1024
    %   Reference: IEEE 802.11
    
    validateattributes(fftSize, {'numeric'}, {"scalar", "integer"}, mfilename, "FFTSize", 1);
    if log2(fftSize) ~= floor(log2(fftSize))
        error("FFT size must be the power of 2");
    end

    switch(fftSize)
        case 32 % IEEE 802.11ah
            nullSCIdx = [-16:-14 0 14:15];
        case 64 % IEEE 802.11ac
            nullSCIdx = [-32:-29 0 29:31];
        case 128 % IEEE 802.11ac
            nullSCIdx = [-64:-59 -1:1 59:63];
        case 256 % IEEE 802.11ax
            nullSCIdx = [-128:-123 -1:1 123:127];
        case 512 % IEEE 802.11ax
            nullSCIdx = [-256:-245 -2:2 245:255];
        case 1024 % IEEE 802.11ax
            nullSCIdx = [-512:-501 -2:-2 501:511];
        otherwise
            error("FFT size is not support: %d", fftSize);
    end

    nullSCIdx = nullSCIdx + fftSize / 2 + 1; % Turn to matlab index
end

