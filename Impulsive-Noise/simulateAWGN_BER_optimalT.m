clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
fftSize = 256;                  % FFT size
dataFactor = [7 1];             % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
pilotIdx = []; pilotVal = [];   % Pilot Subcarrier Index & Values
% nullIdx = getNullIdx(fftSize);  % Null Subcarrier Index
% [pilotIdx, pilotVal] = getPilotIdxAndVal(fftSize); % Pilot Subcarrier Index & Values
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 10000;             % Every loop test signals
ebn0List = 10:1:20;             % Energy per bit to noise power spectral density ratio(dB)

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprob = 0.01;                  % Probability of impulsive noise occurring

% Non-linear process paramter
% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

tClipBlank = @(T) (T * 1.4);    % Clipping-Blanking
deepMu = 0.5;                   % Deep-Clipping
tReplaceClip = @(T) (T * 1.2);  % Replacement-Clipping-Blanking Clipping
tReplaceBlank = @(T) (T * 1.4); % Replacement-Clipping-Blanking Blanking

rng(2025);
gpurng(2025);

%% value depends on parameter
numData = fftSize - length(nullIdx) - length(pilotIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx; pilotIdx]);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;
pilotVal = repmat(pilotVal, 1, sigPerLoop);

% PGIR transform domain mask
transMask = zeros(fftSize, 1);
transMask([nullIdx; pilotIdx]) = 1;

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

%% data storage
berPGIR = zeros(1, length(ebn0List));
berBlank = zeros(1, length(ebn0List));
berClip = zeros(1, length(ebn0List));
berClipBlank = zeros(1, length(ebn0List));
berDeepClip = zeros(1, length(ebn0List));
berReplace = zeros(1, length(ebn0List));
berReplaceClipBlank = zeros(1, length(ebn0List));

%% CP-OFDM
% for idxINprob = 1:length(INprobList)
%     INprob = INprobList(idxINprob);
fprintf('Start process %f probability at %s\n', INprob, datetime('now', TimeZone='local', Format='MM-dd HH:mm:ss'));

for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / fftSize);
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    % BER storage depends on EbN0
    tempBERPGIR = zeros(1, totalSigCount / sigPerLoop);
    tempBERBlank = zeros(1, totalSigCount / sigPerLoop);
    tempBERClip = zeros(1, totalSigCount / sigPerLoop);
    tempBERClipBlank = zeros(1, totalSigCount / sigPerLoop);
    % tempBERDeepClip = zeros(1, totalSigCount / sigPerLoop);
    % tempBERReplace = zeros(1, totalSigCount / sigPerLoop);
    % tempBERReplaceClipBlank = zeros(1, totalSigCount / sigPerLoop);

    sigP = (fftSize - length(nullIdx)) / fftSize;
    noiseP = sigP / 10 ^ (snr / 10);
    INP = sigP / 10 ^ (INsnr / 10);

    optThresPGIR = getOptThresPGIR(sigP, noiseP, INP, INprob, ...
            fftSize, nullIdx, 'nIter', iterCount, 'M', 1000);
    optThresBlank = getOptThresBlanking(sigP, noiseP, INP, INprob);
    optThresClip = getOptThresClipping(sigP, noiseP, INP, INprob);
    optThresClipBlank = getOptThresClipBlanking(sigP, noiseP, INP, INprob);
    
    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);

    parfor idxRun = 1:(totalSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop]);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx, pilotIdx, pilotVal);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);

        sigPower = calcPower(txIFFTSig);
        % sigP = mean(sigPower);

        % Noise
        [noise, noisePower] = awgnx(size(txIFFTSig), snr, sigPower, txIFFTSig(1));
        rxNoisySig = txIFFTSig + noise;

        % Impulsive Noise
        [impulsiveNoise, INPower, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
        rxINNoisySig = rxNoisySig + impulsiveNoise;

        ampThreshold = optThresPGIR;
        dataMask = abs(rxINNoisySig) < ampThreshold;
        rxPGIRSig = PGIR(rxINNoisySig, txIFFTSig, dataMask, transMask, iterCount);
        rxPGIRFDSig = 1 / sqrt(fftSize) .* fft(rxPGIRSig, fftSize, 1);
        rxDemapPGIRSig = scDemap(rxPGIRFDSig, fftSize, nullIdx, pilotIdx);
        outPGIRDataBits = demodulator(rxDemapPGIRSig, modOrder);

        % ampThreshold = optTList(2);
        ampThreshold = optThresBlank;
        rxBlankSig = rxINNoisySig;
        idxBlank = abs(rxBlankSig) > ampThreshold;
        rxBlankSig(idxBlank) = 0;
        rxBlankFDSig = 1 / sqrt(fftSize) .* fft(rxBlankSig, fftSize, 1);
        rxDemapBlankSig = scDemap(rxBlankFDSig, fftSize, nullIdx, pilotIdx);
        outBlankDataBits = demodulator(rxDemapBlankSig, modOrder);

        ampThreshold = optThresClip;
        rxClipSig = rxINNoisySig;
        idxClip = abs(rxClipSig) > ampThreshold;
        rxClipSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipSig(idxClip)));
        rxClipFDSig = 1 / sqrt(fftSize) .* fft(rxClipSig, fftSize, 1);
        rxDemapClipSig = scDemap(rxClipFDSig, fftSize, nullIdx, pilotIdx);
        outClipDataBits = demodulator(rxDemapClipSig, modOrder);

        ampThreshold = optThresClipBlank;
        rxClipBlankSig = rxINNoisySig;
        idxClip = abs(rxClipBlankSig) > ampThreshold;
        idxBlank = abs(rxClipBlankSig) > tClipBlank(ampThreshold);
        rxClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipBlankSig(idxClip)));
        rxClipBlankSig(idxBlank) = 0;
        rxClipBlankFDSig = 1 / sqrt(fftSize) .* fft(rxClipBlankSig, fftSize, 1);
        rxDemapClipBlankSig = scDemap(rxClipBlankFDSig, fftSize, nullIdx, pilotIdx);
        outClipBlankDataBits = demodulator(rxDemapClipBlankSig, modOrder);

        % ampThreshold = optTList(5);
        % rxDeepClipSig = rxINNoisySig;
        % idxDeepClip = abs(rxDeepClipSig) > ampThreshold;
        % idxBlank = abs(rxDeepClipSig) > ((1 + deepMu) / deepMu * ampThreshold);
        % rxDeepClipSig(idxDeepClip) = (ampThreshold - deepMu .* (abs(rxDeepClipSig(idxDeepClip)) - ampThreshold)) ...
        %     .* exp(1j .* angle(rxDeepClipSig(idxDeepClip)));
        % rxDeepClipSig(idxBlank) = 0;
        % rxDeepClipFDSig = 1 / sqrt(fftSize) .* fft(rxDeepClipSig, fftSize, 1);
        % rxDemapDeepClipSig = scDemap(rxDeepClipFDSig, fftSize, nullIdx, pilotIdx);
        % outDeepClipDataBits = demodulator(rxDemapDeepClipSig, modOrder);
        % 
        % ampThreshold = optTList(6);
        % rxReplaceSig = rxINNoisySig;
        % idxReplace = abs(rxReplaceSig) > ampThreshold;
        % rxReplaceSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceSig(idxReplace)));
        % rxReplaceFDSig = 1 / sqrt(fftSize) .* fft(rxReplaceSig, fftSize, 1);
        % rxDemapReplaceSig = scDemap(rxReplaceFDSig, fftSize, nullIdx, pilotIdx);
        % outReplaceDataBits = demodulator(rxDemapReplaceSig, modOrder);
        % 
        % ampThreshold = optTList(7);
        % rxReplaceClipBlankSig = rxINNoisySig;
        % idxClip = abs(rxReplaceClipBlankSig) > ampThreshold;
        % idxReplace = abs(rxReplaceClipBlankSig) > tReplaceClip(ampThreshold);
        % idxBlank = abs(rxReplaceClipBlankSig) > tReplaceBlank(ampThreshold);
        % rxReplaceClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxReplaceClipBlankSig(idxClip)));
        % rxReplaceClipBlankSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceClipBlankSig(idxReplace)));
        % rxReplaceClipBlankSig(idxBlank) = 0;
        % rxReplaceClipBlankFDSig = 1 / sqrt(fftSize) .* fft(rxReplaceClipBlankSig, fftSize, 1);
        % rxDemapReplaceClipBlankSig = scDemap(rxReplaceClipBlankFDSig, fftSize, nullIdx, pilotIdx);
        % outReplaceClipBlankDataBits = demodulator(rxDemapReplaceClipBlankSig, modOrder);
    
        [~, tempBERPGIR(idxRun)] = biterr(inDataBits(:), outPGIRDataBits(:));
        [~, tempBERBlank(idxRun)] = biterr(inDataBits(:), outBlankDataBits(:));
        [~, tempBERClip(idxRun)] = biterr(inDataBits(:), outClipDataBits(:));
        [~, tempBERClipBlank(idxRun)] = biterr(inDataBits(:), outClipBlankDataBits(:));
        % [~, tempBERDeepClip(idxRun)] = biterr(inDataBits(:), outDeepClipDataBits(:));
        % [~, tempBERReplace(idxRun)] = biterr(inDataBits(:), outReplaceDataBits(:));
        % [~, tempBERReplaceClipBlank(idxRun)] = biterr(inDataBits(:), outReplaceClipBlankDataBits(:));
    end

    berPGIR(idxEbn0) = mean(tempBERPGIR);
    berBlank(idxEbn0) = mean(tempBERBlank);
    berClip(idxEbn0) = mean(tempBERClip);
    berClipBlank(idxEbn0) = mean(tempBERClipBlank);
    % berDeepClip(idxEbn0) = mean(tempBERDeepClip);
    % berReplace(idxEbn0) = mean(tempBERReplace);
    % berReplaceClipBlank(idxEbn0) = mean(tempBERReplaceClipBlank);

end

%% plot figure
figure
semilogy(ebn0List, berPGIR, DisplayName=sprintf("PGIR %d times", iterCount));
hold on;
semilogy(ebn0List, berBlank, DisplayName="Blanking");
semilogy(ebn0List, berClip, DisplayName="Clipping");
semilogy(ebn0List, berClipBlank, DisplayName="Clipping + Blanking");
% semilogy(ebn0List, berDeepClip, DisplayName="Deep Clipping");
% semilogy(ebn0List, berReplace, DisplayName="Replacement");
% semilogy(ebn0List, berReplaceClipBlank, DisplayName="Replacement + Clipping + Blanking");
hold off;
legend;
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
grid on;
