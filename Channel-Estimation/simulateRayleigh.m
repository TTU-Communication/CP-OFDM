clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
fftSize = 64;                   % FFT size
nullIdx = getNullIdx(fftSize);                   % Null Subcarrier Index
[pilotIdx, pilotVal] = getPilotIdxAndVal(fftSize);   % Pilot Subcarrier Index & Values
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Channel parameter
channelLen = 8;                 % Multipath length

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 100;               % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
numData = fftSize - length(nullIdx) - length(pilotIdx);     % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx; pilotIdx]);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;
pilotVal = repmat(pilotVal, [1 sigBatchPerLoop]);

sigPowerRef = (numData + length(pilotIdx)) / fftSize;

%% data storage
ber = zeros(1, length(ebn0List));
estBER = zeros(1, length(ebn0List));

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / (fftSize + cpLen));
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    % BER storage depends on EbN0
    tempBER = zeros(1, totalSigCount / sigBatchPerLoop);
    tempEstBER = zeros(1, totalSigCount / sigBatchPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    parfor idxRun = 1:(totalSigCount / sigBatchPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigBatchPerLoop]);
        txModSig = modulator(inDataBits, modOrder, lower(modType));
        txMapSig = scMap(txModSig, fftSize, nullIdx, pilotIdx, pilotVal);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);

        % Channel
        [fadedSig, channel] = rayleighChannel(txOFDMSig, channelLen, 1 / channelLen, fftSize);

        % Noise
        noisePower = sigPowerRef / (10 ^ (snr / 10));
        noise = awgnx(size(fadedSig), noisePower, fadedSig(1));
        rxNoisySig = fadedSig + noise;

        % Rx
        rxNoCPSig = cpRemover(rxNoisySig, cpLen);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);

        % actual channel
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx, pilotIdx);
        rxEQSig = equalizer(rxDemapSig, channel(dataIdx, :));
        outDataBits = demodulator(rxEQSig, modOrder, lower(modType));

        % estimated channel
        [rxEstDemapSig, rxPilotSig] = scDemap(rxFFTSig, fftSize, nullIdx, pilotIdx);
        estChannel = channelEstimator(rxPilotSig ./ pilotVal, fftSize, nullIdx, pilotIdx);
        rxEstEQSig = equalizer(rxEstDemapSig, estChannel);
        estOutDataBits = demodulator(rxEstEQSig, modOrder, lower(modType));

        % BER calculate
        [~, tempBER(idxRun)] = biterr(inDataBits, outDataBits);
        [~, tempEstBER(idxRun)] = biterr(inDataBits, estOutDataBits);

    end

    ber(idxEbn0) = mean(tempBER);
    estBER(idxEbn0) = mean(tempEstBER);

end

%% plot BER
figure
semilogy(ebn0List, ber, DisplayName='actual channel');
hold on;
semilogy(ebn0List, estBER, DisplayName='estimated channel');
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
legend
grid on;
