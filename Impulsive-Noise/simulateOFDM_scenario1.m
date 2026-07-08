clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
fftSize = 256;                   % FFT size
dataFactor = [7 1]; % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
channelLen = 8;                 % Multipath length in rayleight distribution (no LoS)
kFactor = 8;
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')
nTX = 1;
nRX = 1;
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 10000;               % Every loop test signals
ebn0List = 10:1:30;              % Energy per bit to noise power spectral density ratio(dB)
INsnr = -15;
INprob = 0.01;
iterCount = 20;
rng(2026);
gpurng(2026);

%% value depends on parameter
numData = fftSize - length(nullIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

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
cpAdder = @(sig, len) sig([end-len+1:end, 1:end], :, :);
cpRemover = @(sig, len) sig(len+1:end, :, :);

transMask = gpuArray.zeros(fftSize, 1);
transMask(nullIdx) = 1;
transMask = repmat(transMask, 1, sigPerLoop);
sigRef = gpuArray.zeros(fftSize, sigPerLoop);

%% data storage
ber = zeros(1, length(ebn0List));
berIN = zeros(1, length(ebn0List));
berINPGIR = zeros(1, length(ebn0List));

%% CP-OFDM
for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / (fftSize + cpLen));
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    % BER storage depends on EbN0
    tempBER = gpuArray.zeros(1, totalSigCount / sigPerLoop);
    tempBERIN = gpuArray.zeros(1, totalSigCount / sigPerLoop);
    tempBERPGIR = gpuArray.zeros(1, totalSigCount / sigPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    for idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop nTX]);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);

        % Channel
        sigPower = calcPower(txOFDMSig);
        % [fadedSig, channel] = ricianChannel(txOFDMSig, fftSize, channelLen, kFactor, nRX);
        fadedSig = txOFDMSig;
        channel = 1;

        % Noise
        [noise, noisePower] = awgnx(size(fadedSig), snr, sigPower, fadedSig(1));
        rxNoisySig = fadedSig + noise;

        [impulsiveNoise, ~, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
        rxINNoisySig = rxNoisySig + impulsiveNoise;

        % Rx
        rxNoCPSig = cpRemover(rxNoisySig, cpLen);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);
        rxEQSig = equalizer(rxFFTSig, channel);
        rxDemapSig = scDemap(rxEQSig, fftSize, nullIdx);
        outDataBits = demodulator(rxDemapSig, modOrder);

        rxINNoCPSig = cpRemover(rxINNoisySig, cpLen);
        rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINNoCPSig, fftSize, 1);
        rxINEQSig = equalizer(rxINFFTSig, channel);
        rxINDemapSig = scDemap(rxINEQSig, fftSize, nullIdx);
        outINDataBits = demodulator(rxINDemapSig, modOrder);

        dataMask = gpuArray(~happenIdx((cpLen+1):end, :,:,:));
        rxPGIRTDSig = sqrt(fftSize) .* ifft(rxINEQSig, fftSize, 1);
        rxPGIRSig = PGIR(rxPGIRTDSig, sigRef, dataMask, transMask, iterCount);
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

%%
idx = 4;
cpuRxNoCPSig = gather(rxNoCPSig(:, idx));
cpuRxINNoCPSig = gather(rxINNoCPSig(:, idx));
cpuRxDemapSig = gather(rxDemapSig(:, idx));
cpuRxINDemapSig = gather(rxINDemapSig(:, idx));
cpuRxPGIRDemapSig = gather(rxPGIRDemapSig(:, idx));
cpuRxPGIRFDSig = gather(rxPGIRFDSig(:, idx));
cpuRxEQFDSig = gather(rxEQSig(:, idx));
cpuRxINEQFDSig = gather(rxINEQSig(:, idx));
cpuRxPGIRTDSig = gather(rxPGIRSig(:, idx));
cpuRxEQTDSig = sqrt(fftSize) .* ifft(cpuRxEQFDSig);
cpuRxINEQTDSig = sqrt(fftSize) .* ifft(cpuRxINEQFDSig);

figure
hold on;
xtap = 1:size(cpuRxNoCPSig, 1);

stairs(xtap, abs(cpuRxNoCPSig));
stairs(xtap, abs(cpuRxINNoCPSig));
hold off;
ylabel('|y|');
xlabel('Sample');
title('Signal Sample (Before EQ)');
legend('Orig. Signal', 'Orig. + IN Signal');

figure
hold on;
xtap = 1:size(cpuRxEQTDSig, 1);

stairs(xtap, abs(cpuRxEQTDSig));
stairs(xtap, abs(cpuRxINEQTDSig));
stairs(xtap, abs(cpuRxPGIRTDSig));
hold off;
ylabel('|y|');
xlabel('Sample');
title('Signal Sample (After EQ)');
legend('Orig. Signal', 'Orig. + IN Signal', sprintf('PGIR %d time(s)', iterCount));

figure
hold on;
% plot(real(cpuRxDemapSig), imag(cpuRxDemapSig), LineStyle="none", Marker=".", MarkerSize=12);
% plot(real(cpuRxINDemapSig), imag(cpuRxINDemapSig), LineStyle="none", Marker=".", MarkerSize=12);
% plot(real(cpuRxPGIRDemapSig), imag(cpuRxPGIRDemapSig), LineStyle="none", Marker=".", MarkerSize=12);
plot(ifftshift(abs(cpuRxEQFDSig)));
plot(ifftshift(abs(cpuRxINEQFDSig)));
plot(ifftshift(abs(cpuRxPGIRFDSig)));
hold off;
ylabel('|y|');
xlabel('Sample');
title('Spetrum');
legend('Orig. Signal', 'Orig. + IN Signal', sprintf('PGIR %d time(s)', iterCount));