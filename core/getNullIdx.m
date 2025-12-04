function [nullIdx] = getNullIdx(nfft, length, DCLength)
% GETNULLIDX get the correspond null subcarrier indices to fft size
%
%   Avaliable FFT size: 32, 64, 128, 256, 512, 1024
%   Reference: IEEE 802.11
    
    narginchk(1, 3);

    validateattributes(nfft, {'numeric'}, {"scalar", "integer", 'positive'}, mfilename, "NFFT", 1);

    if nargin == 1

        if log2(nfft) ~= floor(log2(nfft))
            error("FFT size must be the power of 2");
        end

        switch(nfft)
            case 32 % IEEE 802.11ah
                negNullIdx = -16:-14;
                posNullIdx = [0 14:15];
            case 64 % IEEE 802.11ac
                negNullIdx = -32:-29;
                posNullIdx = [0 29:31];
            case 128 % IEEE 802.11ac
                negNullIdx = [-64:-59 -1];
                posNullIdx = [0:1 59:63];
            case 256 % IEEE 802.11ax
                negNullIdx = [-128:-123 -1];
                posNullIdx = [0:1 123:127];
            case 512 % IEEE 802.11ax
                negNullIdx = [-256:-245 -2:-1];
                posNullIdx = [0:2 245:255];
            case 1024 % IEEE 802.11ax
                negNullIdx = [-512:-501 -2:-1];
                posNullIdx = [0:2 501:511];
            otherwise
                error("FFT size is not support: %d", nfft);
        end

    else
        validateattributes(length, {'numeric'}, {"scalar", "integer", '<', nfft}, mfilename, "LENGTH", 2);
        if nargin == 3
            validateattributes(DCLength, {'numeric'}, {"scalar", "integer", '<', length}, mfilename, "DCLENGTH", 3);
        else
            DCLength = 0;
        end

        nfftHalf = nfft / 2;
        length = length - DCLength;
        negHalf = ceil(length / 2);
        posHalf = length - negHalf;

        negNullIdx = -nfftHalf:(-nfftHalf + negHalf - 1);
        posNullIdx = (nfftHalf - posHalf):(nfftHalf - 1);

        if DCLength > 0
            DCPosHalf = ceil(DCLength / 2);
            DCNegHalf = DCLength - DCPosHalf;

            negNullIdx = [negNullIdx -DCNegHalf:-1];
            posNullIdx = [0:(DCPosHalf-1) posNullIdx];
        end
    end

    % Turn to matlab index
    nullIdx = [posNullIdx (negNullIdx + nfft)]' + 1;
end

