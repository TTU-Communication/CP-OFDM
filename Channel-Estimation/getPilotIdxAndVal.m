function [pilotIdx, pilotValue] = getPilotIdxAndVal(nfft)
% GETPILOTIDXANDVAL get the correspond pilot subcarrier indices and values
% to fft size
%
%   Avaliable FFT size: 32, 64, 128, 256, 512, 1024
%   Reference: IEEE 802.11
    
    validateattributes(nfft, {'numeric'}, {"scalar", "integer"}, mfilename, "NFFT", 1);
    if log2(nfft) ~= floor(log2(nfft))
        error("FFT size must be the power of 2");
    end

    tempPilotValue = [1 1 1 -1 -1 1 1 1];

    switch(nfft)
        case 32 % IEEE 802.11ah
            pilotIdx = -7;
            pilotValueIdx = 3:4;
        case 64 % IEEE 802.11ac
            pilotIdx = [-21 -7];
            pilotValueIdx = 1:4;
        case 128 % IEEE 802.11ac
            pilotIdx = [-53 -25 -11];
            pilotValueIdx = 1:6;
        case 256 % IEEE 802.11ax
            pilotIdx = [-116 -90 -48 -22];
            pilotValueIdx = 1:8;
        case 512 % IEEE 802.11ax
            pilotIdx = [-238 -212 -170 -144 -104 -78 -36 -10];
            pilotValueIdx = 1:8;
        case 1024 % IEEE 802.11ax
            pilotIdx = [-468 -400 -334 -266 -226 -158 -92 -24];
            pilotValueIdx = 1:8;
        otherwise
            error("FFT size is not support: %d", nfft);
    end

    pilotValue = tempPilotValue(pilotValueIdx)';
    pilotIdx = [pilotIdx -flip(pilotIdx)]' + nfft / 2 + 1; % Turn to matlab index
end

