clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'core')));

%% parameter setting
fftSize = 32;                   % FFT size
nullIdx = [];                   % Null Subcarrier Index
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
channelLen = 8;                 % Multipath length in rayleight distribution (no LoS)
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')
nTX = 2;
nRX = 2;
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 100;               % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
numData = fftSize - length(nullIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

%% package 
randomBits = @(sigSize) randi([0 1], sigSize);
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
cpAdder = @(sig, len) sig([end-len+1:end, 1:end], :, :);
cpRemover = @(sig, len) sig(len+1:end, :, :);

%% data storage
ber = zeros(1, length(ebn0List));

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / (fftSize + cpLen));
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    % BER storage depends on EbN0
    tempBER = zeros(1, totalSigCount / sigPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    parfor idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop nTX]);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);

        % Channel
        sigPower = calcPower(txOFDMSig);
        [fadedSig, channel] = rayleighChannel(txOFDMSig, channelLen, 1/channelLen, fftSize, nRX);

        % Noise
        [noise, noisePower] = awgnx(size(fadedSig), snr, sigPower, fadedSig(1));
        rxNoisySig = fadedSig + noise;

        % Rx
        rxNoCPSig = cpRemover(rxNoisySig, cpLen);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);
        rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);
        rxEQSig = equalizer(rxDemapSig, channel(dataIdx, :, :, :));
        outDataBits = demodulator(rxEQSig, modOrder);

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
