function [pilotSCIdx, pilotValue] = getPilotSCIndexAndValue(fftSize)
    %GETPILOTSCINDEXANDVALUE get the correspond null subcarrier index to fft
    %size
    %   Avaliable FFT size: 64, 128, 256, 512, 1024
    %   Reference: IEEE 802.11
    
    validateattributes(fftSize, {'numeric'}, {"scalar", "integer"}, mfilename, "FFTSize", 1);
    if log2(fftSize) ~= floor(log2(fftSize))
        error("FFT size must be the power of 2");
    end

    switch(fftSize)
        case 64 % IEEE 802.11ac
            pilotSCIdx = [-21 -7];
            pilotValue = [1 1 1 -1].';
        case 128 % IEEE 802.11ac
            pilotSCIdx = [-53 -25 -11];
            pilotValue = [1 1 1 -1 -1 1].';
        case 256 % IEEE 802.11ax
            pilotSCIdx = [-116 -90 -48 -22];
            pilotValue = [1 1 1 -1 -1 1 1 1].';
        case 512 % IEEE 802.11ax
            pilotSCIdx = [-238 -212 -170 -144 -104 -78 -36 -10];
            pilotValue = [1 1 1 -1 -1 1 1 1].';
        case 1024 % IEEE 802.11ax
            pilotSCIdx = [-468 -400 -334 -266 -226 -158 -92 -24];
            pilotValue = [1 1 1 -1 -1 1 1 1].';
        otherwise
            error("FFT size is not support: %d", fftSize);
    end

    pilotSCIdx = [pilotSCIdx -flip(pilotSCIdx)] + fftSize / 2 + 1; % Turn to matlab index
end

