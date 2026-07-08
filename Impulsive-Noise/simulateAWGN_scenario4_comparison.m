clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
fftSize = 256;                  % FFT size
dataFactor = [7 1];             % data, null ratio
nullIdx = getNullIdx(fftSize);  % Null Subcarrier Index
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 10000;             % Every loop test signals
snr = 25;                       % Noise-to-Signal ratio

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprob = 0.01;                  % Probability of impulsive noise occurring

% Non-linear process paramter
ampThresholdList = 0.1:0.1:15;  % Amplitude threshold

% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

tClipBlank = @(T) (T * 1.4);    % Clipping-Blanking
deepMu = 0.5;                   % Deep-Clipping
tReplaceClip = @(T) (T * 1.2);  % Replacement-Clipping-Blanking Clipping
tReplaceBlank = @(T) (T * 1.4); % Replacement-Clipping-Blanking Blanking

rng(2025);
gpurng(2025);

%% value depends on parameter
numData = fftSize - length(nullIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

% PGIR transform domain mask
transMask = gpuArray.zeros(fftSize, 1);
transMask(nullIdx) = 1;

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
pgirSNREff = zeros(1, length(ampThresholdList));
blankSNREff = zeros(1, length(ampThresholdList));
clipSNREff = zeros(1, length(ampThresholdList));
clipblankSNREff = zeros(1, length(ampThresholdList));
deepclipSNREff = zeros(1, length(ampThresholdList));
replaceSNREff = zeros(1, length(ampThresholdList));
replaceclipblankSNREff = zeros(1, length(ampThresholdList));

%% CP-OFDM
% Tx
inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop]);
txModSig = modulator(inDataBits, modOrder);
txMapSig = scMap(txModSig, fftSize, nullIdx);
txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);

% Channel
sigPower = calcPower(txIFFTSig);
sigP = mean(sigPower);

% Noise
[noise, noisePower] = awgnx(size(txIFFTSig), snr, sigPower, txIFFTSig(1));
rxNoisySig = txIFFTSig + noise;

% Impulsive Noise
[impulsiveNoise, ~, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
rxINNoisySig = rxNoisySig + impulsiveNoise;

% Rx
for idxAmpThreshold = 1:length(ampThresholdList)
    ampThreshold = ampThresholdList(idxAmpThreshold);

    dataMask = abs(rxINNoisySig) < ampThreshold;
    rxPGIRSig = PGIR(rxINNoisySig, txIFFTSig, dataMask, transMask, iterCount);
    pgirK_0 = mean(rxPGIRSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    pgirTempSNREff = sum(calcPower(pgirK_0 .* txIFFTSig)) ./ sum(calcPower(rxPGIRSig - pgirK_0 .* txIFFTSig));

    pgirSNREff(idxAmpThreshold) = gather(mean(pgirTempSNREff));

    rxBlankSig = rxINNoisySig;
    idxBlank = abs(rxBlankSig) > ampThreshold;
    rxBlankSig(idxBlank) = 0;
    blankK_0 = mean(rxBlankSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    blankTempSNREff = sum(calcPower(blankK_0 .* txIFFTSig)) ./ sum(calcPower(rxBlankSig - blankK_0 .* txIFFTSig));

    blankSNREff(idxAmpThreshold) = gather(mean(blankTempSNREff));

    rxClipSig = rxINNoisySig;
    idxClip = abs(rxClipSig) > ampThreshold;
    rxClipSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipSig(idxClip)));
    clipK_0 = mean(rxClipSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    clipTempSNREff = sum(calcPower(clipK_0 .* txIFFTSig)) ./ sum(calcPower(rxClipSig - clipK_0 .* txIFFTSig));

    clipSNREff(idxAmpThreshold) = gather(mean(clipTempSNREff));

    rxClipBlankSig = rxINNoisySig;
    idxClip = abs(rxClipBlankSig) > ampThreshold;
    idxBlank = abs(rxClipBlankSig) > tClipBlank(ampThreshold);
    rxClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipBlankSig(idxClip)));
    rxClipBlankSig(idxBlank) = 0;
    clipblankK_0 = mean(rxClipBlankSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    clipblankTempSNREff = sum(calcPower(clipblankK_0 .* txIFFTSig)) ./ sum(calcPower(rxClipBlankSig - clipblankK_0 .* txIFFTSig));

    clipblankSNREff(idxAmpThreshold) = gather(mean(clipblankTempSNREff));

    rxDeepClipSig = rxINNoisySig;
    idxDeepClip = abs(rxDeepClipSig) > ampThreshold;
    idxBlank = abs(rxDeepClipSig) > ((1 + deepMu) / deepMu * ampThreshold);
    rxDeepClipSig(idxDeepClip) = (ampThreshold - deepMu .* (abs(rxDeepClipSig(idxDeepClip)) - ampThreshold)) .* exp(1j .* angle(rxDeepClipSig(idxDeepClip)));
    rxDeepClipSig(idxBlank) = 0;
    deepclipK_0 = mean(rxDeepClipSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    deepclipTempSNREff = sum(calcPower(deepclipK_0 .* txIFFTSig)) ./ sum(calcPower(rxDeepClipSig - deepclipK_0 .* txIFFTSig));

    deepclipSNREff(idxAmpThreshold) = gather(mean(deepclipTempSNREff));

    rxReplaceSig = rxINNoisySig;
    idxReplace = abs(rxReplaceSig) > ampThreshold;
    rxReplaceSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceSig(idxReplace)));
    replaceK_0 = mean(rxReplaceSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
    replaceTempSNREff = sum(calcPower(replaceK_0 .* txIFFTSig)) ./ sum(calcPower(rxReplaceSig - replaceK_0 .* txIFFTSig));

    replaceSNREff(idxAmpThreshold) = gather(mean(replaceTempSNREff));

    rxReplaceclipblankSig = rxINNoisySig;
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

%% plot figure
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
