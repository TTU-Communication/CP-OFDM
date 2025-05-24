function [nullIdx] = getNullIdx(nfft)
% GETNULLIDX get the correspond null subcarrier indices to fft size
%
%   Avaliable FFT size: 32, 64, 128, 256, 512, 1024
%   Reference: IEEE 802.11
    
    validateattributes(nfft, {'numeric'}, {"scalar", "integer"}, mfilename, "NFFT", 1);
    if log2(nfft) ~= floor(log2(nfft))
        error("FFT size must be the power of 2");
    end

    switch(nfft)
        case 32 % IEEE 802.11ah
            nullIdx = [-16:-14 0 14:15];
        case 64 % IEEE 802.11ac
            nullIdx = [-32:-29 0 29:31];
        case 128 % IEEE 802.11ac
            nullIdx = [-64:-59 -1:1 59:63];
        case 256 % IEEE 802.11ax
            nullIdx = [-128:-123 -1:1 123:127];
        case 512 % IEEE 802.11ax
            nullIdx = [-256:-245 -2:2 245:255];
        case 1024 % IEEE 802.11ax
            nullIdx = [-512:-501 -2:-2 501:511];
        otherwise
            error("FFT size is not support: %d", nfft);
    end

    nullIdx = nullIdx' + nfft / 2 + 1; % Turn to matlab index
end

