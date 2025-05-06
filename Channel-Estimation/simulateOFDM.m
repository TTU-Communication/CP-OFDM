clc; clear;

%% parameter setting
fftSize = 64;                   % FFT size
nullIdx = getNullSCIndex(fftSize);  % Null Subcarrier Index
[pilotIdx, pilotValue] = getPilotSCIndexAndValue(fftSize);  % Pilot Subcarrier Index & Value
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
channelLen = 8;                 % Multipath length in rayleight distribution (no LoS)
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')
baseSigCount = 1000;            % testing signal numbers (will multiply a factor)
sigPerLoop = 100;               % Every loop test signals
ebn0List = 0:1:20;              % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
numData = fftSize - length(nullIdx) - length(pilotIdx);     % Data subcarrier size
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
    tempBER = zeros(1, totalSigCount / sigPerLoop);
    tempEstBER = zeros(1, totalSigCount / sigPerLoop);

    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    parfor idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits(bitsPerOFDMSymbol, sigPerLoop);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = subcarrierMapping(txModSig, fftSize, pilotIdx, pilotValue, nullIdx);
        txMapSig = ifftshift(txMapSig, 1);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);

        % Channel
        sigPower = calcPower(txOFDMSig);
        [fadedSig, channel] = rayleighChannel(txOFDMSig, channelLen, 1/channelLen, fftSize);
        channel = fftshift(channel, 1);

        % Noise
        noise = awgnx(size(fadedSig), snr, sigPower, fadedSig(1));
        noisePower = calcPower(noise);
        rxNoisySig = fadedSig + noise;

        % Rx
        rxNoCPSig = cpRemover(rxNoisySig, cpLen);
        rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);
        rxFFTSig = fftshift(rxFFTSig, 1);

        % actual channel
        rxEQSig = equalizer(rxFFTSig, channel);
        rxDemapSig = subcarrierDemapping(rxEQSig, pilotIdx, nullIdx);
        outDataBits = demodulator(rxDemapSig, modOrder);

        % estimated channel
        estChannel = channelEstimator(rxFFTSig, pilotIdx, pilotValue, nullIdx);
        rxEstEQSig = equalizer(rxFFTSig, estChannel);
        rxEstDemapSig = subcarrierDemapping(rxEstEQSig, pilotIdx, nullIdx);
        estOutDataBits = demodulator(rxEstDemapSig, modOrder);

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
