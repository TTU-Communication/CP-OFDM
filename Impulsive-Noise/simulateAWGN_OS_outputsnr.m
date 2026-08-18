clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
numData = 64;                   % Data subcarrier size
OSFactor = 4;                   % oversampling factor
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 10000;        % Every loop test signals
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
tClipReplace = @(T) (T * 1.2);  % Clipping-Replacement-Blanking Clipping
tClReBlank = @(T) (T * 1.4);    % Clipping-Replacement-Blanking Blanking

rng(2025);
gpurng(2025);

%% value depends on parameter
fftSize = numData * OSFactor;
nullIdx = getNullIdx(fftSize, (OSFactor - 1) * numData);
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

sigPowerRef = numData * OSFactor / fftSize;
% Calculate noise power & impulsive noise power
noisePower = sigPowerRef / (10 ^ (snr / 10));
INPower = sigPowerRef / (10 ^ (INsnr / 10));

% PGIR transform domain mask
transMask = gpuArray.zeros(fftSize, 1);
transMask(nullIdx) = 1;
refSig = gpuArray.zeros(fftSize, 1);

%% data storage
pgirSNREff = zeros(length(INprobList), length(ampThresholdList));
blankSNREff = zeros(length(INprobList), length(ampThresholdList));
clipSNREff = zeros(length(INprobList), length(ampThresholdList));
clipBlankSNREff = zeros(length(INprobList), length(ampThresholdList));
deepClipSNREff = zeros(length(INprobList), length(ampThresholdList));
replaceSNREff = zeros(length(INprobList), length(ampThresholdList));
clipReplaceBlankSNREff = zeros(length(INprobList), length(ampThresholdList));

%% CP-OFDM
for idxINprob = 1:length(INprobList)
    INprob = INprobList(idxINprob);
    fprintf('Start process %f probability at %s\n', INprob, getTimeStr);

    % Tx
    inDataBits = randomBits([bitsPerOFDMSymbol*1 1*sigBatchPerLoop], OutputLocation='gpu');
    txModSig = modulator(inDataBits, modOrder, lower(modType));
    txPreMapSig = reshape(txModSig, [numData 1 1 sigBatchPerLoop]);
    txMapSig = scMap(txPreMapSig, fftSize, nullIdx);
    txIFFTSig = sqrt(OSFactor) .* sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
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
        pgirSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxPGIRSig, txIFFTSig));

        rxBlankSig = blanking(rxINSig, ampThreshold);
        blankSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxBlankSig, txIFFTSig));
    
        rxClipSig = clipping(rxINSig, ampThreshold);
        clipSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxClipSig, txIFFTSig));
    
        rxClipBlankSig = clipBlank(rxINSig, ampThreshold, tClipBlank(ampThreshold));
        clipBlankSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxClipBlankSig, txIFFTSig));
    
        rxDeepClipSig = deepClip(rxINSig, ampThreshold, deepMu);
        deepClipSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxDeepClipSig, txIFFTSig));
    
        rxReplaceSig = replacement(rxINSig, ampThreshold, sigPowerRef);
        replaceSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxReplaceSig, txIFFTSig));
    
        rxClipReplaceBlankSig = clipReplaceBlank(rxINSig, ampThreshold, tClipReplace(ampThreshold), ...
                                    tClReBlank(ampThreshold), sigPowerRef);
        clipReplaceBlankSNREff(idxINprob, idxAmpThreshold) = gather(calcOutputSNR(rxClipReplaceBlankSig, txIFFTSig));

    end

end

%% plot figure
linewidth = 1.5;
markersize = 10;
markerSpace = 5;
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
    plot(ampThresholdList, 10 * log10(clipReplaceBlankSNREff(idxINprob, :)), '-', Marker='*', Color=c(idxINprob, :), ...
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
title(sprintf('N = %d, Oversampling %d', fftSize, OSFactor), ...
    Interpreter="latex", FontSize=16);

%%
function [outSNR] = calcOutputSNR(inSig, refSig)
    K_0 = sum(inSig .* conj(refSig), "all") / sum(abs(refSig) .^ 2, "all");
    sigPower = abs(K_0) ^ 2 * sum(abs(refSig) .^ 2, "all");
    errPower = sum(abs(inSig - K_0 .* refSig) .^ 2, "all");

    outSNR = sigPower / errPower;
end
