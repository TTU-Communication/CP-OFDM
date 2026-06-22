clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
fftSize = 256;                   % FFT size
dataFactor = [7 1]; % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
% nullIdx = getNullIdx(fftSize);
pilotIdx = []; pilots = [];
% [pilotIdx, pilots] = getPilotIdxAndVal(fftSize);
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
channelLen = 8;                 % Multipath length in rayleight distribution (no LoS)
kFactor = 0;
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
INprobList = [0.001 0.01 0.1];
iterCount = 20;
ampThresholdList = [0.1:0.1:15];
tClipBlank = @(T) (T * 1.4);
deepMu = 0.5;
tReplaceClip = @(T) (T * 1.2);
tReplaceBlank = @(T) (T * 1.4);
rng(2025);
gpurng(2025);

%% value depends on parameter
numData = fftSize - length(nullIdx) - length(pilotIdx);    % Data subcarrier size
dataIdx = setdiff((1:fftSize)', [nullIdx; pilotIdx]);
pilots = repmat(pilots, 1, sigPerLoop);
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

transMask = zeros(fftSize, sigPerLoop);
transMask([nullIdx; pilotIdx], :) = 1;
sigRefFD = zeros(fftSize, sigPerLoop);
sigRefFD(pilotIdx, :) = pilots;
sigRef = sqrt(fftSize) .* ifft(sigRefFD, fftSize, 1);

%% data storage
% rxPGIRSigList = cell(1, length(iterCountList));
% rxPGIRFDSigList = cell(1, length(iterCountList));
% rxPGIRDemapSigList = cell(1, length(iterCountList));
% nmseList = cell(1, length(iterCountList));
% avgNMSEList = zeros(1, length(iterCountList));
% berINPGIRList = zeros(1, length(iterCountList));
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

    pgirNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    pgirDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    blankNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    blankDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    clipNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    clipDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    clipBlankNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    clipBlankDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    deepClipNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    deepClipDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    replaceNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    replaceDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    replaceClipBlankNum = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));
    replaceClipBlankDen = zeros(baseSigCount / sigPerLoop, length(ampThresholdList));

    for idxLoop = 1:(baseSigCount / sigPerLoop)
        inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop nTX]);
        txModSig = modulator(inDataBits, modOrder);
        txMapSig = scMap(txModSig, fftSize, nullIdx, pilotIdx, pilots);
        txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
        txOFDMSig = cpAdder(txIFFTSig, cpLen);

        % Channel
        sigPower = calcPower(txOFDMSig);
        sigP = mean(sigPower);
        [fadedSig, channel] = ricianChannel(txOFDMSig, channelLen, kFactor, fftSize, nRX);
        % fadedSig = txOFDMSig;
        % channel = 1;

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
        % rxPGIRTDSig = sqrt(fftSize) .* ifft(rxINEQSig, fftSize, 1);
        rxPGIRTDSig = rxINNoCPSig;

        parfor idxAmpThreshold = 1:length(ampThresholdList)
            ampThreshold = ampThresholdList(idxAmpThreshold);

            dataMask = abs(rxPGIRTDSig) < ampThreshold;
            rxPGIRSig = PGIR(rxPGIRTDSig, sigRef, dataMask, transMask, iterCount);
            rxPGIREQTDSig = autoConvertFDDoEQ(rxPGIRSig, fftSize, channel, noisePower);
            pgirK_0 = mean(rxPGIREQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % pgirTempSNREff = sum(calcPower(pgirK_0 .* txIFFTSig)) ./ sum(calcPower(rxPGIREQTDSig - pgirK_0 .* txIFFTSig));
            % 
            % pgirSNREff(idxINprob, idxAmpThreshold) = gather(mean(pgirTempSNREff));
            pgirNum(idxLoop, idxAmpThreshold) = sum(calcPower(pgirK_0 .* txIFFTSig));
            pgirDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxPGIREQTDSig - pgirK_0 .* txIFFTSig));

            rxBlankSig = rxPGIRTDSig;
            idxBlank = abs(rxBlankSig) > ampThreshold;
            rxBlankSig(idxBlank) = 0;
            rxBlankEQTDSig = autoConvertFDDoEQ(rxBlankSig, fftSize, channel, noisePower);
            blankK_0 = mean(rxBlankEQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % blankTempSNREff = sum(calcPower(blankK_0 .* txIFFTSig)) ./ sum(calcPower(rxBlankEQTDSig - blankK_0 .* txIFFTSig));
            % 
            % blankSNREff(idxINprob, idxAmpThreshold) = gather(mean(blankTempSNREff));
            blankNum(idxLoop, idxAmpThreshold) = sum(calcPower(blankK_0 .* txIFFTSig));
            blankDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxBlankEQTDSig - blankK_0 .* txIFFTSig));

            rxClipSig = rxPGIRTDSig;
            idxClip = abs(rxClipSig) > ampThreshold;
            rxClipSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipSig(idxClip)));
            rxClipEQTDSig = autoConvertFDDoEQ(rxClipSig, fftSize, channel, noisePower);
            clipK_0 = mean(rxClipEQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % clipTempSNREff = sum(calcPower(clipK_0 .* txIFFTSig)) ./ sum(calcPower(rxClipEQTDSig - clipK_0 .* txIFFTSig));
            % 
            % clipSNREff(idxINprob, idxAmpThreshold) = gather(mean(clipTempSNREff));
            clipNum(idxLoop, idxAmpThreshold) = sum(calcPower(clipK_0 .* txIFFTSig));
            clipDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxClipEQTDSig - clipK_0 .* txIFFTSig));

            rxClipBlankSig = rxPGIRTDSig;
            idxClip = abs(rxClipBlankSig) > ampThreshold;
            idxBlank = abs(rxClipBlankSig) > tClipBlank(ampThreshold);
            rxClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxClipBlankSig(idxClip)));
            rxClipBlankSig(idxBlank) = 0;
            rxClipBlankEQTDSig = autoConvertFDDoEQ(rxClipBlankSig, fftSize, channel, noisePower);
            clipBlankK_0 = mean(rxClipBlankEQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % clipBlankTempSNREff = sum(calcPower(clipBlankK_0 .* txIFFTSig)) ...
            %     ./ sum(calcPower(rxClipBlankEQTDSig - clipBlankK_0 .* txIFFTSig));
            % 
            % clipBlankSNREff(idxINprob, idxAmpThreshold) = gather(mean(clipBlankTempSNREff));
            clipBlankNum(idxLoop, idxAmpThreshold) = sum(calcPower(clipBlankK_0 .* txIFFTSig));
            clipBlankDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxClipBlankEQTDSig - clipBlankK_0 .* txIFFTSig));

            rxDeepClipSig = rxPGIRTDSig;
            idxDeepClip = abs(rxDeepClipSig) > ampThreshold;
            idxBlank = abs(rxDeepClipSig) > ((1 + deepMu) / deepMu * ampThreshold);
            rxDeepClipSig(idxDeepClip) = (ampThreshold - deepMu .* (abs(rxDeepClipSig(idxDeepClip)) - ampThreshold)) ...
                .* exp(1j .* angle(rxDeepClipSig(idxDeepClip)));
            rxDeepClipSig(idxBlank) = 0;
            rxDeepClipEQTDSig = autoConvertFDDoEQ(rxDeepClipSig, fftSize, channel, noisePower);
            deepClipK_0 = mean(rxDeepClipEQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % deepClipTempSNREff = sum(calcPower(deepClipK_0 .* txIFFTSig)) ...
            %     ./ sum(calcPower(rxDeepClipEQTDSig - deepClipK_0 .* txIFFTSig));
            % 
            % deepClipSNREff(idxINprob, idxAmpThreshold) = gather(mean(deepClipTempSNREff));
            deepClipNum(idxLoop, idxAmpThreshold) = sum(calcPower(deepClipK_0 .* txIFFTSig));
            deepClipDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxDeepClipEQTDSig - deepClipK_0 .* txIFFTSig));

            rxReplaceSig = rxPGIRTDSig;
            idxReplace = abs(rxReplaceSig) > ampThreshold;
            rxReplaceSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceSig(idxReplace)));
            rxReplaceEQTDSig = autoConvertFDDoEQ(rxReplaceSig, fftSize, channel, noisePower);
            replaceK_0 = mean(rxReplaceEQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % replaceTempSNREff = sum(calcPower(replaceK_0 .* txIFFTSig)) ...
            %     ./ sum(calcPower(rxReplaceEQTDSig - replaceK_0 .* txIFFTSig));
            % 
            % replaceSNREff(idxINprob, idxAmpThreshold) = gather(mean(replaceTempSNREff));
            replaceNum(idxLoop, idxAmpThreshold) = sum(calcPower(replaceK_0 .* txIFFTSig));
            replaceDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxReplaceEQTDSig - replaceK_0 .* txIFFTSig));

            rxReplaceClipBlankSig = rxPGIRTDSig;
            idxClip = abs(rxReplaceClipBlankSig) > ampThreshold;
            idxReplace = abs(rxReplaceClipBlankSig) > tReplaceClip(ampThreshold);
            idxBlank = abs(rxReplaceClipBlankSig) > tReplaceBlank(ampThreshold);
            rxReplaceClipBlankSig(idxClip) = ampThreshold .* exp(1j .* angle(rxReplaceClipBlankSig(idxClip)));
            rxReplaceClipBlankSig(idxReplace) = (sqrt(pi .* sigP ./ 4)) .* exp(1j .* angle(rxReplaceClipBlankSig(idxReplace)));
            rxReplaceClipBlankSig(idxBlank) = 0;
            rxReplaceClipBlankEQTDSig = autoConvertFDDoEQ(rxReplaceClipBlankSig, fftSize, channel, noisePower);
            replaceClipBlankK_0 = mean(rxReplaceClipBlankEQTDSig .* conj(txIFFTSig), 1) ./ calcPower(txIFFTSig);
            % replaceClipBlankTempSNREff = sum(calcPower(replaceClipBlankK_0 .* txIFFTSig)) ...
            %     ./ sum(calcPower(rxReplaceClipBlankEQTDSig - replaceClipBlankK_0 .* txIFFTSig));
            % 
            % replaceClipBlankSNREff(idxINprob, idxAmpThreshold) = gather(mean(replaceClipBlankTempSNREff));
            replaceClipBlankNum(idxLoop, idxAmpThreshold) = sum(calcPower(replaceClipBlankK_0 .* txIFFTSig));
            replaceClipBlankDen(idxLoop, idxAmpThreshold) = sum(calcPower(rxReplaceClipBlankEQTDSig - replaceClipBlankK_0 .* txIFFTSig));

        end

    end

    pgirSNREff(idxINprob, :) = sum(pgirNum, 1) ./ sum(pgirDen, 1);
    blankSNREff(idxINprob, :) = sum(blankNum, 1) ./ sum(blankDen, 1);
    clipSNREff(idxINprob, :) = sum(clipNum, 1) ./ sum(clipDen, 1);
    clipBlankSNREff(idxINprob, :) = sum(clipBlankNum, 1) ./ sum(clipBlankDen, 1);
    deepClipSNREff(idxINprob, :) = sum(deepClipNum, 1) ./ sum(deepClipDen, 1);
    replaceSNREff(idxINprob, :) = sum(replaceNum, 1) ./ sum(replaceDen, 1);
    replaceClipBlankSNREff(idxINprob, :) = sum(replaceClipBlankNum, 1) ./ sum(replaceClipBlankDen, 1);

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
title(sprintf('$\\|I_T^\\prime\\|_0:\\|I_T\\|_0=%d:%d$, Rayleigh', dataFactor(1), dataFactor(2)), ...
    Interpreter="latex", FontSize=17);

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
title(sprintf('$\\|I_T^\\prime\\|_0:\\|I_T\\|_0=%d:%d$', dataFactor(1), dataFactor(2)), ...
    Interpreter="latex", FontSize=16);

%%
function [afterEQTDSig] = autoConvertFDDoEQ(inSig, fftSize, channel, noisePower)
    inFDSig = 1 / sqrt(fftSize) .* fft(inSig, fftSize, 1);
    afterEQSig = equalizer(inFDSig, channel, noisePower);
    afterEQTDSig = sqrt(fftSize) .* ifft(afterEQSig, fftSize, 1);
end