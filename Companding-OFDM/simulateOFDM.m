clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
fftSize = 2048;                 % FFT size
nullIdx = getNullIdx(fftSize, 2048 - 640, 1);  % Null Subcarrier Index
cpLen = fftSize * 1 / 8;        % Cyclic Prefix size
% channelLen = 8;                 % Multipath length in rayleight distribution (no LoS)
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 100;               % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

mu = 1;                         % Companding parameter (mu-law)

%% value depends on parameter
numData = fftSize - length(nullIdx);                  % Data subcarrier size
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

%% package 
randomBits = @(r, c) randi([0 1], r, c);
calcPower = @(sig) sum(abs(sig) .^ 2) / size(sig, 1);
switch (lower(modType))
    case 'psk'
        modulator = @(input, M) pskmod(input, M, InputType="bit");
        demodulator = @(input, M) pskdemod(input, M, OutputType="bit");
    case 'qam'
        modulator = @(input, M) qammod(input, M, InputType="bit", ...
            UnitAveragePower=true);
        demodulator = @(input, M) qamdemod(input, M, OutputType="bit", ...
            UnitAveragePower=true);
    otherwise
        error('OFDMMain:invalidModulation', ...
            'The modulation mode must be one of PSK or QAM.');
end
cpAdder = @(sig, len) [sig(end-len+1:end, :); sig];
cpRemover = @(sig, len) sig(len+1:end, :);
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
        + 10 * log10(numData / (fftSize + cpLen));
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    % BER storage depends on EbN0
    tempBER = zeros(1, totalSigCount / sigPerLoop);
    tempBERCmp = zeros(1, totalSigCount / sigPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    if idxEbn0 == length(ebn0List)
        paprCP = zeros(totalSigCount / sigPerLoop, sigPerLoop);
        paprCmp = zeros(totalSigCount / sigPerLoop, sigPerLoop);
        randIdx = randi([1 totalSigCount / sigPerLoop], 1);
    end

    parfor idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits(bitsPerOFDMSymbol, sigPerLoop);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);
        txCmpSig = companding(txOFDMSig, mu);

        % Channel CP-OFDM
        sigPower = calcPower(txOFDMSig);
        % [fadedSig, channel] = rayleighChannel(txOFDMSig, channelLen, 1/channelLen, fftSize);
        fadedSig = txOFDMSig;
        % Channel Companding OFDM
        cmpSigPower = calcPower(txCmpSig);
        % [fadedCmpSig, channelCmp] = rayleighChannel(txCmpSig, channelLen, 1/channelLen, fftSize);
        fadedCmpSig = txCmpSig;

        % Noise CP-OFDM
        noise = awgnx(size(fadedSig), snr, sigPower, fadedSig(1));
        noisePower = calcPower(noise);
        rxNoisySig = fadedSig + noise;
        % Noise Companding OFDM
        noiseCmp = awgnx(size(fadedCmpSig), snr, cmpSigPower, fadedCmpSig(1));
        noiseCmpPower = calcPower(noiseCmp);
        rxNoisyCmpSig = fadedCmpSig + noiseCmp;

        % Rx CP-OFDM
        rxNoCPSig = cpRemover(rxNoisySig, cpLen);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);
        % rxEQSig = equalizer(rxDemapSig, channel(setdiff(1:fftSize, nullIdx), :));
        rxEQSig = rxDemapSig;
        outDataBits = demodulator(rxEQSig, modOrder);
        % Rx Companding OFDM
        rxDeCmpSig = decompanding(rxNoisyCmpSig, mu);
        rxNoCPCmpSig = cpRemover(rxDeCmpSig, cpLen);
        rxFFTCmpSig = 1 / sqrt(fftSize) .* fft(rxNoCPCmpSig, fftSize, 1);
        rxDemapCmpSig = scDemap(rxFFTCmpSig, fftSize, nullIdx);
        % rxEQCmpSig = equalizer(rxFFTCmpSig, channelCmp(setdiff(1:fftSize, nullIdx), :));
        rxEQCmpSig = rxDemapCmpSig;
        outDataBitsCmp = demodulator(rxEQCmpSig, modOrder);
        
        % BER calculate
        [~, tempBER(idxRun)] = biterr(inDataBits, outDataBits);
        [~, tempBERCmp(idxRun)] = biterr(inDataBits, outDataBitsCmp);

        if idxEbn0 == length(ebn0List)
            paprCP(idxRun, :) = peakAvgDB(txOFDMSig);
            paprCmp(idxRun, :) = peakAvgDB(txCmpSig);
            if idxRun == randIdx
                cpSigSample = [cpSigSample; txOFDMSig(:, 1)];
                cmpSigSample = [cmpSigSample; txCmpSig(:, 1)];
            end

        end

    end

    ber(idxEbn0) = mean(tempBER);
    berCmp(idxEbn0) = mean(tempBERCmp);

end

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
