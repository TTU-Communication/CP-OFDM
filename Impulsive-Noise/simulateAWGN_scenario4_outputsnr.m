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
sigBatchPerLoop = 10000;             % Every loop test signals
snr = 25;                       % Noise-to-Signal ratio

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprobList = [0.001 0.01 0.1];  % Probability of impulsive noise occurring

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
numData = fftSize - length(nullIdx) - length(pilotIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx; pilotIdx]);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;
pilotVal = repmat(pilotVal, [1 1 1 sigBatchPerLoop]);

sigPowerRef = (numData + length(pilotIdx)) / fftSize;

% PGIR transform domain mask
transMask = gpuArray.zeros(fftSize, 1);
transMask([nullIdx; pilotIdx]) = 1;
refSigFD = gpuArray.zeros(fftSize, 1, 1, sigBatchPerLoop);
refSigFD(pilotIdx, :, :, :) = pilotVal;
refSig = sqrt(fftSize) * ifft(refSigFD, fftSize, 1);

%% data storage
pgirSNREff = zeros(length(INprobList), length(ampThresholdList));
blankSNREff = zeros(length(INprobList), length(ampThresholdList));
clipSNREff = zeros(length(INprobList), length(ampThresholdList));
clipBlankSNREff = zeros(length(INprobList), length(ampThresholdList));
deepClipSNREff = zeros(length(INprobList), length(ampThresholdList));
replaceSNREff = zeros(length(INprobList), length(ampThresholdList));
replaceClipBlankSNREff = zeros(length(INprobList), length(ampThresholdList));

%% CP-OFDM
for idxINprob = 1:length(INprobList)
    INprob = INprobList(idxINprob);

    fprintf('Start process %f probability at %s\n', INprob, getTimeStr);

    % Calculate noise power & impulsive noise power
    noisePower = sigPowerRef / (10 ^ (snr / 10));
    INPower = sigPowerRef / (10 ^ (INsnr / 10));

    % Tx
    inDataBits = randomBits([bitsPerOFDMSymbol*1 1*sigBatchPerLoop], OutputLocation='gpu');
    txModSig = modulator(inDataBits, modOrder, lower(modType));
    txPreMapSig = reshape(txModSig, [numData 1 1 sigBatchPerLoop]);
    txMapSig = scMap(txPreMapSig, fftSize, nullIdx, pilotIdx, pilotVal);
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
        rxPGIRSig = PGIR(rxINSig, txIFFTSig, dataMask, transMask, iterCount);
        pgirSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxPGIRSig, txIFFTSig));

        rxBlankSig = rxINSig;
        idxBlank = abs(rxBlankSig) > ampThreshold;
        rxBlankSig(idxBlank) = 0;
        blankSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxBlankSig, txIFFTSig));

        rxClipSig = rxINSig;
        idxClip = abs(rxClipSig) > ampThreshold;
        rxClipSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipSig(idxClip)));
        clipSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxClipSig, txIFFTSig));

        rxClipBlankSig = rxINSig;
        idxClip = abs(rxClipBlankSig) > ampThreshold;
        idxBlank = abs(rxClipBlankSig) > tClipBlank(ampThreshold);
        rxClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipBlankSig(idxClip)));
        rxClipBlankSig(idxBlank) = 0;
        clipBlankSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxClipBlankSig, txIFFTSig));

        rxDeepClipSig = rxINSig;
        idxDeepClip = abs(rxDeepClipSig) > ampThreshold;
        idxBlank = abs(rxDeepClipSig) > ((1 + deepMu) / deepMu * ampThreshold);
        rxDeepClipSig(idxDeepClip) = (ampThreshold - deepMu .* (abs(rxDeepClipSig(idxDeepClip)) - ampThreshold)) ...
            .* exp(1j .* angle(rxDeepClipSig(idxDeepClip)));
        rxDeepClipSig(idxBlank) = 0;
        deepClipSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxDeepClipSig, txIFFTSig));

        rxReplaceSig = rxINSig;
        idxReplace = abs(rxReplaceSig) > ampThreshold;
        rxReplaceSig(idxReplace) = (sqrt(pi .* sigPowerRef ./ 4)) .* exp(1j .* angle(rxReplaceSig(idxReplace)));
        replaceSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxReplaceSig, txIFFTSig));

        rxReplaceClipBlankSig = rxINSig;
        idxClip = abs(rxReplaceClipBlankSig) > ampThreshold;
        idxReplace = abs(rxReplaceClipBlankSig) > tReplaceClip(ampThreshold);
        idxBlank = abs(rxReplaceClipBlankSig) > tReplaceBlank(ampThreshold);
        rxReplaceClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxReplaceClipBlankSig(idxClip)));
        rxReplaceClipBlankSig(idxReplace) = (sqrt(pi .* sigPowerRef ./ 4)) .* exp(1j .* angle(rxReplaceClipBlankSig(idxReplace)));
        rxReplaceClipBlankSig(idxBlank) = 0;
        replaceClipBlankSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxReplaceClipBlankSig, txIFFTSig));

    end

end

%% plot figure
linewidth = 1.5;
markersize = 10;
markerSpace = 3;
markerIdx = markerSpace:markerSpace:length(ampThresholdList);
c = orderedcolors('gem12');

figure;
hold on; grid on;  box on;

for idxINprob = 1:length(INprobList)
    plot(ampThresholdList, 10 * log10(pgirSNREff(idxINprob, :)), '-', Color=c(idxINprob, :), ...
        DisplayName='PGIR', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(blankSNREff(idxINprob, :)), '--', Color=c(idxINprob, :), ...
        DisplayName='Blanking', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(clipSNREff(idxINprob, :)), '-.', Color=c(idxINprob, :), ...
        DisplayName='Clipping', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(clipBlankSNREff(idxINprob, :)), '-', Marker='o', Color=c(idxINprob, :), ...
        DisplayName='Clipping + Blanking', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(deepClipSNREff(idxINprob, :)), '-', Marker='+', Color=c(idxINprob, :), ...
        DisplayName='Deep Clipping', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(replaceSNREff(idxINprob, :)), '-', Marker='x', Color=c(idxINprob, :), ...
        DisplayName='Replacement', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(replaceClipBlankSNREff(idxINprob, :)), '-', Marker='*', Color=c(idxINprob, :), ...
        DisplayName='Clipping + Replacement + Blanking', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);

end

nColor = length(INprobList);
nStyle = 7;

colorTexts = {"p=0.001", "p=0.01", "p=0.1"};
styleTexts = {sprintf("PGIR %d time(s)", iterCount), "Blanking", "Clipping", "Clipping + Blanking", "Deep Clipping", ...
                "Replacement", "Clipping + Replacement + Blanking"};
styleLines = {'-', '--', '-.', '-', '-', '-', '-',};
styleMarkers = {'none', 'none', 'none', 'o', '+', 'x', '*'};

hColors = cell(nColor, 1);
hStyles = cell(nStyle, 1);

for idx = 1:nColor
    hColors{idx} = plot(nan, nan, LineStyle="-", Marker="none", LineWidth=1.5, ...
        Color=c(idx,:), DisplayName=colorTexts{idx});
end

for idx = 1:nStyle
    hStyles{idx} = plot(nan, nan, Color="k", LineWidth=1.5, ...
        LineStyle=styleLines{idx}, Marker=styleMarkers{idx}, MarkerSize=8, ...
        displayname=styleTexts{idx});
end

hold off;
legend([hColors{:}, hStyles{:}]);
ylim([-25 snr]);
xlabel('Unknown Data Thershold (T)');
ylabel('Output SNR (\gamma), dB');
title(sprintf('$\\|I_T^\\prime\\|_0:\\|I_T\\|_0=%d:%d$', dataFactor(1), dataFactor(2)), ...
    Interpreter="latex", FontSize=16);

%%
function [outSNR] = calcOutputSNR(inSig, refSig)
    K_0 = sum(inSig .* conj(refSig), "all") / sum(abs(refSig) .^ 2, "all");
    sigPower = abs(K_0) ^ 2 * sum(abs(refSig) .^ 2, "all");
    errPower = sum(abs(inSig - K_0 .* refSig) .^ 2, "all");

    outSNR = sigPower / errPower;
end
