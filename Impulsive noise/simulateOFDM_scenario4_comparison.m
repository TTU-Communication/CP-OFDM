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
% ebn0 = 16;              % Energy per bit to noise power spectral density ratio(dB)
snr = 25;
inCount = 2;
INsnr = -15;
INprob = 0.01;
iterCount = 50;
ampThresholdList = [0.1:0.1:15];
tClipBlank = @(T) (T * 1.4);
deepMu = 0.5;
tReplaceClip = @(T) (T * 1.2);
tReplaceBlank = @(T) (T * 1.4);
rng(2025);
gpurng(2025);

%% value depends on parameter
numData = fftSize - length(nullIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx]);
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
% rxPGIRSigList = cell(1, length(iterCountList));
% rxPGIRFDSigList = cell(1, length(iterCountList));
% rxPGIRDemapSigList = cell(1, length(iterCountList));
% nmseList = cell(1, length(iterCountList));
% avgNMSEList = zeros(1, length(iterCountList));
% berINPGIRList = zeros(1, length(iterCountList));
pgirSNREff = zeros(1, length(ampThresholdList));
blankSNREff = zeros(1, length(ampThresholdList));
clipSNREff = zeros(1, length(ampThresholdList));
clipblankSNREff = zeros(1, length(ampThresholdList));
deepclipSNREff = zeros(1, length(ampThresholdList));
replaceSNREff = zeros(1, length(ampThresholdList));
replaceclipblankSNREff = zeros(1, length(ampThresholdList));

%% CP-OFDM
inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop nTX]);
txModSig = modulator(inDataBits, modOrder);
txMapSig = scMap(txModSig, fftSize, nullIdx);
txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
txOFDMSig = cpAdder(txIFFTSig, cpLen);

% Channel
sigPower = calcPower(txOFDMSig);
sigP = mean(sigPower);
% [fadedSig, channel] = ricianChannel(txOFDMSig, fftSize, channelLen, kFactor, nRX);
fadedSig = txOFDMSig;
channel = 1;

% Noise
[noise, noisePower] = awgnx(size(fadedSig), snr, sigPower, fadedSig(1));
rxNoisySig = fadedSig + noise;

% INsnrLinear = sigPower ./ (10 ^ (INsnr / 10));
% impulsiveNoise = 1 / 2 * sqrt(INsnrLinear) .* (rand(size(rxNoisySig)) + 1j * rand(size(rxNoisySig)));
% orgIndex = randi([1 fftSize], inCount, sigPerLoop);
% index = sub2ind(size(rxNoisySig), orgIndex + cpLen, repmat(1:sigPerLoop, inCount, 1));
% invIndex = setdiff(1:(numel(rxNoisySig)), index);
% impulsiveNoise(invIndex) = 0;
[impulsiveNoise, ~, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
rxINNoisySig = rxNoisySig + impulsiveNoise;

% Orig + IN
rxINNoCPSig = cpRemover(rxINNoisySig, cpLen);
rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINNoCPSig, fftSize, 1);
rxINEQSig = equalizer(rxINFFTSig, channel);

% Orig + IN with PGIR
% rxIndex = sub2ind(size(rxINFFTSig), orgIndex, repmat(1:sigPerLoop, inCount, 1));
% dataMask = ones(fftSize, sigPerLoop);
% dataMask(rxIndex) = 0;
% dataMask = gpuArray(~happenIdx((cpLen+1):end, :,:,:));
rxPGIRTDSig = sqrt(fftSize) .* ifft(rxINEQSig, fftSize, 1);

for idxAmpThreshold = 1:length(ampThresholdList)
    ampThreshold = ampThresholdList(idxAmpThreshold);

    dataMask = abs(rxPGIRTDSig) < ampThreshold;
    rxPGIRSig = PGIR(rxPGIRTDSig, sigRef, dataMask, transMask, iterCount);
    pgirK_0 = mean(rxPGIRSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    pgirTempSNREff = sum(calcPower(pgirK_0 .* txIFFTSig)) ./ sum(calcPower(rxPGIRSig - pgirK_0 .* txIFFTSig));

    pgirSNREff(idxAmpThreshold) = gather(mean(pgirTempSNREff));

    rxBlankSig = rxPGIRTDSig;
    idxBlank = abs(rxBlankSig) > ampThreshold;
    rxBlankSig(idxBlank) = 0;
    blankK_0 = mean(rxBlankSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    blankTempSNREff = sum(calcPower(blankK_0 .* txIFFTSig)) ./ sum(calcPower(rxBlankSig - blankK_0 .* txIFFTSig));

    blankSNREff(idxAmpThreshold) = gather(mean(blankTempSNREff));

    rxClipSig = rxPGIRTDSig;
    idxClip = abs(rxClipSig) > ampThreshold;
    rxClipSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipSig(idxClip)));
    clipK_0 = mean(rxClipSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    clipTempSNREff = sum(calcPower(clipK_0 .* txIFFTSig)) ./ sum(calcPower(rxClipSig - clipK_0 .* txIFFTSig));

    clipSNREff(idxAmpThreshold) = gather(mean(clipTempSNREff));

    rxClipBlankSig = rxPGIRTDSig;
    idxClip = abs(rxClipBlankSig) > ampThreshold;
    idxBlank = abs(rxClipBlankSig) > tClipBlank(ampThreshold);
    rxClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipBlankSig(idxClip)));
    rxClipBlankSig(idxBlank) = 0;
    clipblankK_0 = mean(rxClipBlankSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    clipblankTempSNREff = sum(calcPower(clipblankK_0 .* txIFFTSig)) ./ sum(calcPower(rxClipBlankSig - clipblankK_0 .* txIFFTSig));

    clipblankSNREff(idxAmpThreshold) = gather(mean(clipblankTempSNREff));

    rxDeepClipSig = rxPGIRTDSig;
    idxDeepClip = abs(rxDeepClipSig) > ampThreshold;
    idxBlank = abs(rxDeepClipSig) > ((1 + deepMu) / deepMu * ampThreshold);
    rxDeepClipSig(idxDeepClip) = (ampThreshold - deepMu .* (abs(rxDeepClipSig(idxDeepClip)) - ampThreshold)) .* exp(1j .* angle(rxDeepClipSig(idxDeepClip)));
    rxDeepClipSig(idxBlank) = 0;
    deepclipK_0 = mean(rxDeepClipSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    deepclipTempSNREff = sum(calcPower(deepclipK_0 .* txIFFTSig)) ./ sum(calcPower(rxDeepClipSig - deepclipK_0 .* txIFFTSig));

    deepclipSNREff(idxAmpThreshold) = gather(mean(deepclipTempSNREff));

    rxReplaceSig = rxPGIRTDSig;
    idxReplace = abs(rxReplaceSig) > ampThreshold;
    rxReplaceSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceSig(idxReplace)));
    replaceK_0 = mean(rxReplaceSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    replaceTempSNREff = sum(calcPower(replaceK_0 .* txIFFTSig)) ./ sum(calcPower(rxReplaceSig - replaceK_0 .* txIFFTSig));

    replaceSNREff(idxAmpThreshold) = gather(mean(replaceTempSNREff));

    rxReplaceclipblankSig = rxPGIRTDSig;
    idxClip = abs(rxReplaceclipblankSig) > ampThreshold;
    idxReplace = abs(rxReplaceclipblankSig) > tReplaceClip(ampThreshold);
    idxBlank = abs(rxReplaceclipblankSig) > tReplaceBlank(ampThreshold);
    rxReplaceclipblankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxReplaceclipblankSig(idxClip)));
    rxReplaceclipblankSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceclipblankSig(idxReplace)));
    rxReplaceclipblankSig(idxBlank) = 0;
    replaceclipblankK_0 = mean(rxReplaceclipblankSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    replaceclipblankTempSNREff = sum(calcPower(replaceclipblankK_0 .* txIFFTSig)) ./ sum(calcPower(rxReplaceclipblankSig - replaceclipblankK_0 .* txIFFTSig));

    replaceclipblankSNREff(idxAmpThreshold) = gather(mean(replaceclipblankTempSNREff));

end

%%
idx = 2;
c = orderedcolors('gem12');

figure
hold on;
plot(ampThresholdList, 10 * log10(pgirSNREff), '-', Color=c(idx, :), DisplayName='PGIR');
plot(ampThresholdList, 10 * log10(blankSNREff), '--', Color=c(idx, :), DisplayName='Blanking');
plot(ampThresholdList, 10 * log10(clipSNREff), '-.', Color=c(idx, :), DisplayName='Clipping');
plot(ampThresholdList, 10 * log10(clipblankSNREff), '-', Marker='o', Color=c(idx, :), DisplayName='Clipping + Blanking');
plot(ampThresholdList, 10 * log10(deepclipSNREff), '-', Marker='+', Color=c(idx, :), DisplayName='Deep Clipping');
plot(ampThresholdList, 10 * log10(replaceSNREff), '-', Marker='x', Color=c(idx, :), DisplayName='Replacement');
plot(ampThresholdList, 10 * log10(replaceclipblankSNREff), '-', Marker='*', Color=c(idx, :), DisplayName='Clipping + Replacement + Blanking');
hold off;
legend;
xlabel('Unknown Data Thershold (T)');
ylabel('Output SNR (\gamma), dB');
title(sprintf('p = %f', INprob));
grid on;
