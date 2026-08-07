clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
fftSize = 2048;                 % FFT size
nullIdx = getNullIdx(fftSize, 2048 - 640, 1);   % Null Subcarrier Index
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 100;          % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

% Companding parameter
mu = 1;                         % Companding parameter (mu-law)

%% value depends on parameter
numData = fftSize - length(nullIdx);                  % Data subcarrier size
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

sigPowerRef = numData / fftSize;

%% package 
calcPower = @(sig) sum(abs(sig) .^ 2, "all") / numel(sig);
peakAvgDB = @(sig) 10 * log10(max(abs(sig) .^ 2) ./ mean(abs(sig) .^ 2));

%% data storage
ber = zeros(1, length(ebn0List));
berCmp = zeros(1, length(ebn0List));
cpSigSample = [];
cmpSigSample = [];

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / fftSize);
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);
    % fprintf('Start at %s\n', getTimeStr);

    % BER storage depends on EbN0
    tempBER = zeros(1, totalSigCount / sigBatchPerLoop);
    tempBERCmp = zeros(1, totalSigCount / sigBatchPerLoop);

    if idxEbn0 == length(ebn0List)
        paprCP = zeros(totalSigCount / sigBatchPerLoop, sigBatchPerLoop);
        paprCmp = zeros(totalSigCount / sigBatchPerLoop, sigBatchPerLoop);
        randIdx = randi([1 totalSigCount / sigBatchPerLoop], 1);
    end

    parfor idxRun = 1:(totalSigCount / sigBatchPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigBatchPerLoop]);
        txModSig = modulator(inDataBits, modOrder, lower(modType));
        txMapSig = scMap(txModSig, fftSize, nullIdx);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txCmpSig = companding(txIFFTSig, mu);

        sigPower = calcPower(txIFFTSig);
        cmpSigPower = calcPower(txCmpSig);

        % Noise CP-OFDM
        noisePower = sigPower / (10 ^ (snr / 10));
        noise = awgnx(size(txIFFTSig), noisePower, txIFFTSig(1));
        rxNoisySig = txIFFTSig + noise;
        % Noise Companding OFDM
        noiseCmpPower = cmpSigPower / (10 ^ (snr / 10));
        noiseCmp = awgnx(size(txCmpSig), noiseCmpPower, txCmpSig(1));
        rxNoisyCmpSig = txCmpSig + noiseCmp;

        % Rx CP-OFDM
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoisySig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);
        outDataBits = demodulator(rxDemapSig, modOrder, lower(modType));
        % Rx Companding OFDM
        rxDeCmpSig = decompanding(rxNoisyCmpSig, mu);
        rxFFTCmpSig = 1 / sqrt(fftSize) .* fft(rxDeCmpSig, fftSize, 1);
        rxDemapCmpSig = scDemap(rxFFTCmpSig, fftSize, nullIdx);
        outDataBitsCmp = demodulator(rxDemapCmpSig, modOrder, lower(modType));
        
        % BER calculate
        [~, tempBER(idxRun)] = biterr(inDataBits, outDataBits);
        [~, tempBERCmp(idxRun)] = biterr(inDataBits, outDataBitsCmp);

        if idxEbn0 == length(ebn0List)
            paprCP(idxRun, :) = peakAvgDB(txIFFTSig);
            paprCmp(idxRun, :) = peakAvgDB(txCmpSig);
            if idxRun == randIdx
                cpSigSample = [cpSigSample; txIFFTSig(:, 1)];
                cmpSigSample = [cmpSigSample; txCmpSig(:, 1)];
            end

        end

    end

    ber(idxEbn0) = mean(tempBER);
    berCmp(idxEbn0) = mean(tempBERCmp);

end

% fprintf('\n');
% fprintf('%s\n', repmat('-', 1, 50));
% fprintf('All process has done at %s\n', getTimeStr);

%% plot BER
figure
semilogy(ebn0List, ber, DisplayName='CP OFDM');
hold on;
semilogy(ebn0List, berCmp, DisplayName='Companding OFDM');
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
grid on;
legend;

%% plot PAPR
paprCP = reshape(paprCP, [], 1);
paprCmp = reshape(paprCmp, [], 1);

figure
[ecdfCP, xCP] = ecdf(paprCP);
ccdfCP = 1 - ecdfCP;
semilogy(xCP, ccdfCP, DisplayName='CP OFDM');
grid on; hold on;
[ecdfCmp, xCmp] = ecdf(paprCmp);
ccdfCmp = 1 - ecdfCmp;
semilogy(xCmp, ccdfCmp, DisplayName='Companding OFDM');
legend;
xlim([0 14]);
xlabel('Power (dB)');
ylabel('Probability');
title('PAPR');

%% plot PSD
figure
pspectrum(cpSigSample);
grid on; hold on;
pspectrum(cmpSigSample);
legend('CP OFDM', 'Companding OFDM');

%% plot the line of companding
figure
x = 0:0.01:1;
plot(x, x, DisplayName='y = x');
grid on; hold on;
plot(x, companding(x, mu), DisplayName='Companding');
legend;
xlabel('V_{in}');
ylabel('V_{out}');
title('Conversion curve');
