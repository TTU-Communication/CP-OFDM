clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'core')));

%% parameter setting
% Signal parameter
fftSize = 32;                   % FFT size
nullIdx = [];                   % Null Subcarrier Index
pilotIdx = []; pilotVal = [];   % Pilot Subcarrier Index & Values
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Channel parameter
channelLen = 8;                 % Multipath length
kFactor = 10;                   % LoS vs NLoS power factor
nTX = 1;                        % Numel of transmitter antenna
nRX = 1;                        % Numel of receiver antenna

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 100;          % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
numData = fftSize - length(nullIdx) - length(pilotIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx; pilotIdx]);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;
pilotVal = repmat(pilotVal, [1 1 nTX sigBatchPerLoop]);

sigPowerRef = (numData + length(pilotIdx)) / fftSize;

%% data storage
ber = zeros(1, length(ebn0List));

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(nTX) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / (fftSize + cpLen));
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);
    % fprintf('Start at %s\n', getTimeStr);

    % Calculate noise power
    noisePower = sigPowerRef / (10 ^ (snr / 10));
    % BER storage depends on EbN0
    tempBER = zeros(1, totalSigCount / sigBatchPerLoop);

    for idxRun = 1:(totalSigCount / sigBatchPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol*1 nTX*sigBatchPerLoop]);
        txModSig = modulator(inDataBits, modOrder);
        txModSig = reshape(txModSig, [numData 1 nTX sigBatchPerLoop]);
        txMapSig = scMap(txModSig, fftSize, nullIdx, pilotIdx, pilotVal);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen) / sqrt(nTX);
        txOFDMSig = reshape(txOFDMSig, [fftSize+cpLen*1 nTX sigBatchPerLoop]);

        % Channel
        [fadedSig, channel] = ricianChannel(txOFDMSig, channelLen, kFactor, fftSize, nRX);

        % Noise
        noise = awgnx(size(fadedSig), noisePower, fadedSig(1));
        rxNoisySig = fadedSig + noise;

        % Rx
        rxSig = reshape(rxNoisySig, [fftSize+cpLen 1 nRX sigBatchPerLoop]);
        rxNoCPSig = cpRemover(rxSig, cpLen);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx, pilotIdx);
        rxEQSig = equalizer(rxDemapSig, channel(dataIdx, :, :, :) / sqrt(nTX));
        rxEQSig = reshape(rxEQSig, [numData*1 nTX*sigBatchPerLoop]);
        outDataBits = demodulator(rxEQSig, modOrder);

        % BER calculate
        [~, tempBER(idxRun)] = biterr(inDataBits(:), outDataBits(:));

    end

    ber(idxEbn0) = mean(tempBER);

end

% fprintf('\n');
% fprintf('%s\n', repmat('-', 1, 50));
% fprintf('All process has done at %s\n', getTimeStr);

%% plot BER
figure
semilogy(ebn0List, ber);
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
grid on;
