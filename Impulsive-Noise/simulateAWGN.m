clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
fftSize = 256;                  % FFT size
dataFactor = [7 1];             % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 10000;        % Every loop test signals
ebn0List = 10:1:30;             % Energy per bit to noise power spectral density ratio(dB)

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
% INCount = 2;                    % Number of impulsive noise occurring
INprob = 0.01;                  % Probability of impulsive noise occurring

% Non-linear process paramter
ampThreshold = 1.5;             % Amplitude threshold

% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

rng(2025);
gpurng(2025);

%% value depends on parameter
numData = fftSize - length(nullIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

sigPowerRef = numData / fftSize;

% PGIR transform domain mask
transMask = gpuArray.zeros(fftSize, 1);
transMask(nullIdx) = 1;
refSig = gpuArray.zeros(fftSize, 1);

%% data storage
ber = zeros(1, length(ebn0List));
berIN = zeros(1, length(ebn0List));
berINPGIR = zeros(1, length(ebn0List));

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / fftSize);
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);
    fprintf('Start at %s\n', getTimeStr);

    % Calculate noise power & impulsive noise power
    noisePower = sigPowerRef / (10 ^ (snr / 10));
    INPower = sigPowerRef / (10 ^ (INsnr / 10));
    % BER storage depends on EbN0
    tempBER = gpuArray.zeros(1, totalSigCount / sigBatchPerLoop);
    tempBERIN = gpuArray.zeros(1, totalSigCount / sigBatchPerLoop);
    tempBERPGIR = gpuArray.zeros(1, totalSigCount / sigBatchPerLoop);

    for idxRun = 1:(totalSigCount / sigBatchPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol*1 1*sigBatchPerLoop], OutputLocation='gpu');
        txModSig = modulator(inDataBits, modOrder, lower(modType));
        txPreMapSig = reshape(txModSig, [numData 1 1 sigBatchPerLoop]);
        txMapSig = scMap(txPreMapSig, fftSize, nullIdx);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txSig = reshape(txIFFTSig, [fftSize*1 1 sigBatchPerLoop]);

        % Noise
        noise = awgnx(size(txSig), noisePower, txSig(1));
        rxNoisySig = txSig + noise;

        % impulsiveNoise = 1 / 2 * sqrt(INPower) .* (rand(size(rxNoisySig)) + 1j * rand(size(rxNoisySig)));
        % orgIndex = randi([1 fftSize], INCount, 1, 1, sigBatchPerLoop);
        % index = sub2ind(size(rxNoisySig), orgIndex + cpLen, repmat(1:sigBatchPerLoop, INCount, 1));
        % invIndex = setdiff(1:(numel(rxNoisySig)), index);
        % impulsiveNoise(invIndex) = 0;
        [impulsiveNoise, happenIdx] = IN(size(rxNoisySig), INPower, INprob, rxNoisySig(1));
        rxINNoisySig = rxNoisySig + impulsiveNoise;

        % Rx
        % Orig
        rxSig = reshape(rxNoisySig, [fftSize 1 1 sigBatchPerLoop]);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxSig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);
        rxPreDemodSig = reshape(rxDemapSig, [numData*1 1*sigBatchPerLoop]);
        outDataBits = demodulator(rxPreDemodSig, modOrder, lower(modType));

        % Orig + IN
        rxINSig = reshape(rxINNoisySig, [fftSize 1 1 sigBatchPerLoop]);
        rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINSig, fftSize, 1);
        rxINDemapSig = scDemap(rxINFFTSig, fftSize, nullIdx);
        rxINPreDemodSig = reshape(rxINDemapSig, [numData*1 1*sigBatchPerLoop]);
        outINDataBits = demodulator(rxINPreDemodSig, modOrder, lower(modType));

        % Orig + IN with PGIR
        % rxIndex = sub2ind(size(rxINNoisySig), orgIndex, 1:sigBatchPerLoop);
        % dataMask = ones(fftSize, sigBatchPerLoop);
        % dataMask(rxIndex) = 0;
        % dataMask = gpuArray(~happenIdx((cpLen+1):end, :,:,:));
        dataMask = abs(rxINSig) < ampThreshold;
        rxPGIRSig = PGIR(rxINSig, refSig, dataMask, transMask, iterCount);
        rxPGIRFDSig = 1 / sqrt(fftSize) .* fft(rxPGIRSig, fftSize, 1);
        rxPGIRDemapSig = scDemap(rxPGIRFDSig, fftSize, nullIdx);
        rxPGIRPreDemodSig = reshape(rxPGIRDemapSig, [numData*1 1*sigBatchPerLoop]);
        outINPGIRDataBits = demodulator(rxPGIRPreDemodSig, modOrder, lower(modType));

        % BER calculate
        [~, tempBER(idxRun)] = biterr(inDataBits(:), outDataBits(:));
        [~, tempBERIN(idxRun)] = biterr(inDataBits(:), outINDataBits(:));
        [~, tempBERPGIR(idxRun)] = biterr(inDataBits(:), outINPGIRDataBits(:));

    end

    ber(idxEbn0) = mean(tempBER);
    berIN(idxEbn0) = mean(tempBERIN);
    berINPGIR(idxEbn0) = mean(tempBERPGIR);

end

%% plot BER
figure
semilogy(ebn0List, ber);
hold on;
semilogy(ebn0List, berIN);
semilogy(ebn0List, berINPGIR);
hold off;
legend('OFDM', 'OFDM + IN', sprintf('OFDM + IN + PGIR(%d time(s))', iterCount));
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
grid on;
