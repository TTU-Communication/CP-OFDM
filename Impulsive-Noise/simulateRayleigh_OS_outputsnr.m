clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
numData = 64;                   % Data subcarrier size
OSFactor = 4;                   % oversampling factor
cpLenFactor = 1/4;              % The factor for calc. CP from FFT Size
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Channel parameter
channelLen = 8;                 % Multipath length
nTX = 1;                        % Numel of transmitter antenna
nRX = 1;                        % Numel of receiver antenna

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 10000;             % Every loop test signals
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

% rng(2025);
% gpurng(2025);

%% value depends on parameter
fftSize = numData * OSFactor;
cpLen = fftSize * cpLenFactor;
nullIdx = getNullIdx(fftSize, (OSFactor - 1) * numData);
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

% PGIR transform domain mask
transMask = zeros(fftSize, 1);
transMask(nullIdx) = 1;

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

    fprintf('Start process %f probability at %s\n', INprob, datetime('now', TimeZone='local', Format='MM-dd HH:mm:ss'));

    sumTxIFFT2 = 0;
    sumRxPGIRTxIFFT = zeros(1, length(ampThresholdList));
    sumRxPGIR2 = zeros(1, length(ampThresholdList));
    sumRxBlankTxIFFT = zeros(1, length(ampThresholdList));
    sumRxBlank2 = zeros(1, length(ampThresholdList));
    sumRxClipTxIFFT = zeros(1, length(ampThresholdList));
    sumRxClip2 = zeros(1, length(ampThresholdList));
    sumRxClipBlankTxIFFT = zeros(1, length(ampThresholdList));
    sumRxClipBlank2 = zeros(1, length(ampThresholdList));
    sumRxDeepClipTxIFFT = zeros(1, length(ampThresholdList));
    sumRxDeepClip2 = zeros(1, length(ampThresholdList));
    sumRxReplaceTxIFFT = zeros(1, length(ampThresholdList));
    sumRxReplace2 = zeros(1, length(ampThresholdList));
    sumRxReplaceClipBlankTxIFFT = zeros(1, length(ampThresholdList));
    sumRxReplaceClipBlank2 = zeros(1, length(ampThresholdList));

    for idxLoop = 1:(baseSigCount / sigPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop nTX]);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx);
        txIFFTSig = sqrt(OSFactor) .* sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);

        sumTxIFFT2 = sumTxIFFT2 + sum(abs(txIFFTSig).^2, 'all');

        % Channel
        sigPower = calcPower(txOFDMSig);
        sigP = mean(sigPower);
        [fadedSig, channel] = rayleighChannel(txOFDMSig, channelLen, 1 / channelLen, fftSize, nRX);

        % Noise
        [noise, noisePower] = awgnx(size(fadedSig), snr, sigPower, fadedSig(1));
        rxNoisySig = fadedSig + noise;

        % Impulsive Noise
        [impulsiveNoise, ~, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
        rxINNoisySig = rxNoisySig + impulsiveNoise;

        % Rx
        % Orig + IN
        rxINNoCPSig = cpRemover(rxINNoisySig, cpLen);

        for idxAmpThreshold = 1:length(ampThresholdList)
            ampThreshold = ampThresholdList(idxAmpThreshold);

            dataMask = abs(rxINNoCPSig) < ampThreshold;
            rxPGIRSig = PGIR(rxINNoCPSig, txIFFTSig, dataMask, transMask, iterCount);
            rxPGIREQTDSig = autoConvertFDDoEQ(rxPGIRSig, fftSize, OSFactor, channel, noisePower);
            sumRxPGIRTxIFFT(idxAmpThreshold) = sumRxPGIRTxIFFT(idxAmpThreshold) + sum(rxPGIREQTDSig .* conj(txIFFTSig), 'all');
            sumRxPGIR2(idxAmpThreshold) = sumRxPGIR2(idxAmpThreshold) + sum(abs(rxPGIREQTDSig) .^ 2, 'all');

            rxBlankSig = rxINNoCPSig;
            idxBlank = abs(rxBlankSig) > ampThreshold;
            rxBlankSig(idxBlank) = 0;
            rxBlankEQTDSig = autoConvertFDDoEQ(rxBlankSig, fftSize, OSFactor, channel, noisePower);
            sumRxBlankTxIFFT(idxAmpThreshold) = sumRxBlankTxIFFT(idxAmpThreshold) + sum(rxBlankEQTDSig .* conj(txIFFTSig), 'all');
            sumRxBlank2(idxAmpThreshold) = sumRxBlank2(idxAmpThreshold) + sum(abs(rxBlankEQTDSig) .^ 2, 'all');

            rxClipSig = rxINNoCPSig;
            idxClip = abs(rxClipSig) > ampThreshold;
            rxClipSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipSig(idxClip)));
            rxClipEQTDSig = autoConvertFDDoEQ(rxClipSig, fftSize, OSFactor, channel, noisePower);
            sumRxClipTxIFFT(idxAmpThreshold) = sumRxClipTxIFFT(idxAmpThreshold) + sum(rxClipEQTDSig .* conj(txIFFTSig), 'all');
            sumRxClip2(idxAmpThreshold) = sumRxClip2(idxAmpThreshold) + sum(abs(rxClipEQTDSig) .^ 2, 'all');

            rxClipBlankSig = rxINNoCPSig;
            idxClip = abs(rxClipBlankSig) > ampThreshold;
            idxBlank = abs(rxClipBlankSig) > tClipBlank(ampThreshold);
            rxClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipBlankSig(idxClip)));
            rxClipBlankSig(idxBlank) = 0;
            rxClipBlankEQTDSig = autoConvertFDDoEQ(rxClipBlankSig, fftSize, OSFactor, channel, noisePower);
            sumRxClipBlankTxIFFT(idxAmpThreshold) = sumRxClipBlankTxIFFT(idxAmpThreshold) ...
                                                    + sum(rxClipBlankEQTDSig .* conj(txIFFTSig), 'all');
            sumRxClipBlank2(idxAmpThreshold) = sumRxClipBlank2(idxAmpThreshold) + sum(abs(rxClipBlankEQTDSig) .^ 2, 'all');

            rxDeepClipSig = rxINNoCPSig;
            idxDeepClip = abs(rxDeepClipSig) > ampThreshold;
            idxBlank = abs(rxDeepClipSig) > ((1 + deepMu) / deepMu * ampThreshold);
            rxDeepClipSig(idxDeepClip) = (ampThreshold - deepMu .* (abs(rxDeepClipSig(idxDeepClip)) - ampThreshold)) ...
                .* exp(1j .* angle(rxDeepClipSig(idxDeepClip)));
            rxDeepClipSig(idxBlank) = 0;
            rxDeepClipEQTDSig = autoConvertFDDoEQ(rxDeepClipSig, fftSize, OSFactor, channel, noisePower);
            sumRxDeepClipTxIFFT(idxAmpThreshold) = sumRxDeepClipTxIFFT(idxAmpThreshold) ...
                                                   + sum(rxDeepClipEQTDSig .* conj(txIFFTSig), 'all');
            sumRxDeepClip2(idxAmpThreshold) = sumRxDeepClip2(idxAmpThreshold) + sum(abs(rxDeepClipEQTDSig) .^ 2, 'all');

            rxReplaceSig = rxINNoCPSig;
            idxReplace = abs(rxReplaceSig) > ampThreshold;
            rxReplaceSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceSig(idxReplace)));
            rxReplaceEQTDSig = autoConvertFDDoEQ(rxReplaceSig, fftSize, OSFactor, channel, noisePower);
            sumRxReplaceTxIFFT(idxAmpThreshold) = sumRxReplaceTxIFFT(idxAmpThreshold) ...
                                                  + sum(rxReplaceEQTDSig .* conj(txIFFTSig), 'all');
            sumRxReplace2(idxAmpThreshold) = sumRxReplace2(idxAmpThreshold) + sum(abs(rxReplaceEQTDSig) .^ 2, 'all');

            rxReplaceClipBlankSig = rxINNoCPSig;
            idxClip = abs(rxReplaceClipBlankSig) > ampThreshold;
            idxReplace = abs(rxReplaceClipBlankSig) > tReplaceClip(ampThreshold);
            idxBlank = abs(rxReplaceClipBlankSig) > tReplaceBlank(ampThreshold);
            rxReplaceClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxReplaceClipBlankSig(idxClip)));
            rxReplaceClipBlankSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceClipBlankSig(idxReplace)));
            rxReplaceClipBlankSig(idxBlank) = 0;
            rxReplaceClipBlankEQTDSig = autoConvertFDDoEQ(rxReplaceClipBlankSig, fftSize, OSFactor, channel, noisePower);
            sumRxReplaceClipBlankTxIFFT(idxAmpThreshold) = sumRxReplaceClipBlankTxIFFT(idxAmpThreshold) ...
                                                           + sum(rxReplaceClipBlankEQTDSig .* conj(txIFFTSig), 'all');
            sumRxReplaceClipBlank2(idxAmpThreshold) = sumRxReplaceClipBlank2(idxAmpThreshold) ...
                                                      + sum(abs(rxReplaceClipBlankEQTDSig) .^ 2, 'all');

        end

    end

    pgirSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxPGIRTxIFFT, sumRxPGIR2);
    blankSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxBlankTxIFFT, sumRxBlank2);
    clipSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxClipTxIFFT, sumRxClip2);
    clipBlankSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxClipBlankTxIFFT, sumRxClipBlank2);
    deepClipSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxDeepClipTxIFFT, sumRxDeepClip2);
    replaceSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxReplaceTxIFFT, sumRxReplace2);
    replaceClipBlankSNREff(idxINprob, :) = calcBussgangPoolSNR(sumTxIFFT2, sumRxReplaceClipBlankTxIFFT, sumRxReplaceClipBlank2);

end

%%

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

ylim([-25 25]);
xlabel('Unknown Data Thershold (T)');
ylabel('Output SNR (\gamma), dB');
title(sprintf('N = %d, Oversampling %d, Rayleigh, L = %d', fftSize, OSFactor, channelLen), ...
    Interpreter="latex", FontSize=16);

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
ylim([-25 25]);
xlabel('Unknown Data Thershold (T)');
ylabel('Output SNR (\gamma), dB');
title(sprintf('N = %d, Oversampling %d, Rayleigh, L = %d', fftSize, OSFactor, channelLen), ...
    Interpreter="latex", FontSize=16);

%%
function [afterEQTDSig] = autoConvertFDDoEQ(inSig, fftSize, osFactor, channel, noisePower)
    inFDSig = 1 / (sqrt(osFactor) .* sqrt(fftSize)) .* fft(inSig, fftSize, 1);
    afterEQSig = equalizer(inFDSig, channel, noisePower / osFactor);
    afterEQTDSig = sqrt(osFactor) .* sqrt(fftSize) .* ifft(afterEQSig, fftSize, 1);
end

function [outputSNR] = calcBussgangPoolSNR(sumX2, sumYX, sumY2)
    KPool = sumYX ./ sumX2;

    sigPower = abs(KPool).^2 .* sumX2;

    % Equivalent to sum(abs(Y - Kpool*X).^2, 'all'), but more efficient.
    errPower = real(sumY2 - abs(sumYX).^2 ./ sumX2);
    errPower = max(errPower, eps);

    outputSNR = sigPower ./ errPower;
end