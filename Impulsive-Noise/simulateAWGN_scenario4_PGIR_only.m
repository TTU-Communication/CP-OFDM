clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
fftSize = 256;                  % FFT size
dataFactor = [7 1];             % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
pilotIdx = []; pilotVal = [];
% nullIdx = getNullIdx(fftSize);  % Null Subcarrier Index
% [pilotIdx, pilotVal] = getPilotIdxAndVal(fftSize); % Pilot Subcarrier Index & Values
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 10000;             % Every loop test signals
ebn0List = 10:1:30;             % Energy per bit to noise power spectral density ratio(dB)

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprob = 0.01;                  % Probability of impulsive noise occurring

% Non-linear process paramter
ampThreshold = 1.5;             % Amplitude threshold

% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

rng(2025);
gpurng(2025);

%% value depends on parameter
numData = fftSize - length(nullIdx) - length(pilotIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx; pilotIdx]);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;
pilotVal = repmat(pilotVal, 1, sigPerLoop);

% PGIR transform domain mask
transMask = gpuArray.zeros(fftSize, 1);
transMask([nullIdx; pilotIdx]) = 1;

%% package 
randomBits = @(sigSize) gpuArray.randi([0 1], sigSize);
calcPower = @(sig) sum(sum(abs(sig) .^ 2) / size(sig, 1), 3);
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
    % BER storage depends on EbN0
    tempBER = gpuArray.zeros(1, totalSigCount / sigPerLoop);
    tempBERIN = gpuArray.zeros(1, totalSigCount / sigPerLoop);
    tempBERPGIR = gpuArray.zeros(1, totalSigCount / sigPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    for idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop]);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx, pilotIdx, pilotVal);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);

        sigPower = calcPower(txIFFTSig);

        % Noise
        [noise, noisePower] = awgnx(size(txIFFTSig), snr, sigPower, txIFFTSig(1));
        rxNoisySig = txIFFTSig + noise;

        % Impulsive Noise
        [impulsiveNoise, ~, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
        rxINNoisySig = rxNoisySig + impulsiveNoise;

        % Rx
        % Orig
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoisySig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);
        outDataBits = demodulator(rxDemapSig, modOrder);

        % Orig + IN
        rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINNoisySig, fftSize, 1);
        rxINDemapSig = scDemap(rxINFFTSig, fftSize, nullIdx);
        outINDataBits = demodulator(rxINDemapSig, modOrder);

        % Orig + IN with PGIR
        dataMask = gpuArray(~happenIdx);
        % dataMask = abs(rxINNoisySig) < ampThreshold;
        rxPGIRSig = PGIR(rxINNoisySig, txIFFTSig, dataMask, transMask, iterCount);
        rxPGIRFDSig = 1 / sqrt(fftSize) .* fft(rxPGIRSig, fftSize, 1);
        rxPGIRDemapSig = scDemap(rxPGIRFDSig, fftSize, nullIdx);
        outINPGIRDataBits = demodulator(rxPGIRDemapSig, modOrder);

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
