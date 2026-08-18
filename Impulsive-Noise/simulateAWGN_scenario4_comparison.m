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
sigBatchPerLoop = 10000;        % Every loop test signals
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
tClipReplace = @(T) (T * 1.2);  % Clipping-Replacement-Blanking Clipping
tClReBlank = @(T) (T * 1.4);    % Clipping-Replacement-Blanking Blanking

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
pgirSNREff = zeros(1, length(ampThresholdList));
blankSNREff = zeros(1, length(ampThresholdList));
clipSNREff = zeros(1, length(ampThresholdList));
clipBlankSNREff = zeros(1, length(ampThresholdList));
deepClipSNREff = zeros(1, length(ampThresholdList));
replaceSNREff = zeros(1, length(ampThresholdList));
clipReplaceBlankSNREff = zeros(1, length(ampThresholdList));

%% CP-OFDM
% Calculate noise power & impulsive noise power
noisePower = sigPowerRef / (10 ^ (snr / 10));
INPower = sigPowerRef / (10 ^ (INsnr / 10));

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

% Impulsive Noise
[impulsiveNoise, happenIdx] = IN(size(rxNoisySig), INPower, INprob, rxNoisySig(1));
rxINNoisySig = rxNoisySig + impulsiveNoise;

% Rx
rxINSig = reshape(rxINNoisySig, [fftSize 1 1 sigBatchPerLoop]);
for idxAmpThreshold = 1:length(ampThresholdList)
    ampThreshold = ampThresholdList(idxAmpThreshold);

    dataMask = abs(rxINSig) < ampThreshold;
    rxPGIRSig = PGIR(rxINSig, refSig, dataMask, transMask, iterCount);
    pgirSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxPGIRSig, txIFFTSig));

    rxBlankSig = blanking(rxINSig, ampThreshold);
    blankSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxBlankSig, txIFFTSig));

    rxClipSig = clipping(rxINSig, ampThreshold);
    clipSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxClipSig, txIFFTSig));

    rxClipBlankSig = clipBlank(rxINSig, ampThreshold, tClipBlank(ampThreshold));
    clipBlankSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxClipBlankSig, txIFFTSig));

    rxDeepClipSig = deepClip(rxINSig, ampThreshold, deepMu);
    deepClipSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxDeepClipSig, txIFFTSig));

    rxReplaceSig = replacement(rxINSig, ampThreshold, sigPowerRef);
    replaceSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxReplaceSig, txIFFTSig));

    rxClipReplaceBlankSig = clipReplaceBlank(rxINSig, ampThreshold, tClipReplace(ampThreshold), ...
                                tClReBlank(ampThreshold), sigPowerRef);
    clipReplaceBlankSNREff(idxAmpThreshold) = gather(calcOutputSNR(rxClipReplaceBlankSig, txIFFTSig));

end

%% plot figure
idx = 2;
c = orderedcolors('gem12');

figure
hold on;
plot(ampThresholdList, 10 * log10(pgirSNREff), '-', Color=c(idx, :), DisplayName='PGIR');
plot(ampThresholdList, 10 * log10(blankSNREff), '--', Color=c(idx, :), DisplayName='Blanking');
plot(ampThresholdList, 10 * log10(clipSNREff), '-.', Color=c(idx, :), DisplayName='Clipping');
plot(ampThresholdList, 10 * log10(clipBlankSNREff), '-', Marker='o', Color=c(idx, :), DisplayName='Clipping + Blanking');
plot(ampThresholdList, 10 * log10(deepClipSNREff), '-', Marker='+', Color=c(idx, :), DisplayName='Deep Clipping');
plot(ampThresholdList, 10 * log10(replaceSNREff), '-', Marker='x', Color=c(idx, :), DisplayName='Replacement');
plot(ampThresholdList, 10 * log10(clipReplaceBlankSNREff), '-', Marker='*', Color=c(idx, :), DisplayName='Clipping + Replacement + Blanking');
hold off;
legend;
xlabel('Unknown Data Thershold (T)');
ylabel('Output SNR (\gamma), dB');
title(sprintf('p = %f', INprob));
grid on;

%%
function [outSNR] = calcOutputSNR(inSig, refSig)
    K_0 = sum(inSig .* conj(refSig), "all") / sum(abs(refSig) .^ 2, "all");
    sigPower = abs(K_0) ^ 2 * sum(abs(refSig) .^ 2, "all");
    errPower = sum(abs(inSig - K_0 .* refSig) .^ 2, "all");

    outSNR = sigPower / errPower;
end
