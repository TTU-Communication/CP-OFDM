clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'core')));

%% parameter setting
% Signal parameter
fftSize = 32;                   % FFT size
nullIdx = [];                   % Null Subcarrier Index
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 100;               % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
numData = fftSize - length(nullIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

sigPowerRef = numData / fftSize;

%% data storage
ber = zeros(1, length(ebn0List));

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / fftSize);
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    % BER storage depends on EbN0
    tempBER = zeros(1, totalSigCount / sigPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    parfor idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop]);
        txModSig = modulator(inDataBits, modOrder, lower(modType));
        txMapSig = scMap(txModSig, fftSize, nullIdx);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);

        % Noise
        noisePower = sigPowerRef / (10 ^ (snr / 10));
        noise = awgnx(size(txIFFTSig), noisePower, txIFFTSig(1));
        rxNoisySig = txIFFTSig + noise;

        % Rx
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoisySig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);
        outDataBits = demodulator(rxDemapSig, modOrder, lower(modType));

        % BER calculate
        [~, tempBER(idxRun)] = biterr(inDataBits(:), outDataBits(:));

    end

    ber(idxEbn0) = mean(tempBER);

end

%% plot BER
figure
semilogy(ebn0List, ber);
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
grid on;
