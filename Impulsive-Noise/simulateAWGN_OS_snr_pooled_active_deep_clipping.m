clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
numData = 256;                   % Data subcarrier size
OSFactor = 4;                   % oversampling factor
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
Msample = 10000;
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 10000;        % Every loop test signals
snr = 25;                       % Noise-to-Signal ratio

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprob = 0.1;                   % Probability of impulsive noise occurring

% Non-linear process paramter
ampThresholdList = 0.5:0.05:5;  % Amplitude threshold

% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

tClipBlank = @(T) (T * 1.4);    % Clipping-Blanking
deepMuList = [1:0.5:3];                   % Deep-Clipping
tClipReplace = @(T) (T * 1.2);  % Replacement-Clipping-Blanking Clipping
tClReBlank = @(T) (T * 1.4);    % Replacement-Clipping-Blanking Blanking

processName = 'Deep Clipping';

% rng(2025);
% gpurng(2025);

%% value depends on parameter
fftSize = numData * OSFactor;
nullIdx = getNullIdx(fftSize, (OSFactor - 1) * numData);
dataIdx = setdiff((1:fftSize)', nullIdx);
bitsPerModSymbol = log2(modOrder);
bitsPerOFDMSymbol = numData * bitsPerModSymbol;

sigPowerRef = numData * OSFactor / fftSize;

% PGIR transform domain mask
transMask = gpuArray.zeros(fftSize, 1);
transMask(nullIdx) = 1;
refSig = gpuArray.zeros(fftSize, 1);

sigConfig = struct('numData', numData, 'OSFactor', OSFactor, 'modOrder', modOrder, ...
    'modType', modType, 'fftSize', fftSize, 'nullIdx', nullIdx, 'transMask', transMask, ...
    'refSig', refSig, 'sigPowerRef', sigPowerRef, 'bitsPerOFDMSymbol', bitsPerOFDMSymbol);

%% data storage
% K_PGIR_pooled = zeros(length(INprobList), length(ampThresholdList));
% K_PGIR_active = zeros(length(INprobList), length(ampThresholdList));
% snr_PGIR_pooled = zeros(length(INprobList), length(ampThresholdList));
% snr_PGIR_active = zeros(length(INprobList), length(ampThresholdList));
% betaPGIR = zeros(length(INprobList), length(ampThresholdList));
% detCountPGIR = zeros(length(INprobList), length(ampThresholdList));
% berPGIR = zeros(length(INprobList), length(ampThresholdList));

allData = dictionary;
dataSize = [1 length(ampThresholdList)];

for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    switch processName
        case 'PGIR'
            rxHandle = @(config, inSig, thres) rxPGIR1(config, inSig, thres, iterCount);
        case 'Blanking'
            rxHandle = @rxBlank;
        case 'Clipping'
            rxHandle = @rxClip;
        case 'Clipping + Blanking'
            rxHandle = @(config, inSig, thres) rxClipBlank(config, inSig, thres, tClipBlank);
        case 'Deep Clipping'
            rxHandle = @(config, inSig, thres) rxDeepClip(config, inSig, thres, deepMu);
        case 'Replacement'
            rxHandle = @(config, inSig, thres) rxReplace(config, inSig, thres, sigPowerRef);
        case 'Clipping + Replacement + Blanking'
            rxHandle = @(config, inSig, thres) rxClipReplaceBlank(config, inSig, thres, ...
                                                sigPowerRef, tClipReplace, tClReBlank);
        otherwise
            error("Don't have `%s` process. Please check.", processName);
    end
    allData(name) = constructData(dataSize, rxHandle);
end

%% CP-OFDM
fprintf('Start calculate output SNR and others.\n');

fprintf('Start process %f probability at %s\n', INprob, getTimeStr);

[rxINNoisySig, txIFFTSig, txMapSig, txModSig, inDataBits] = makeFrame(sigConfig, Msample, snr, INsnr, INprob);

% Rx
for idxAmpThreshold = 1:length(ampThresholdList)
    ampThreshold = ampThresholdList(idxAmpThreshold);

    for idxDeepMu = 1:length(deepMuList)
        deepMu = deepMuList(idxDeepMu);
        name = processName + string(deepMu);
        [rxSig, rxFDSig, rxDemapSig] = allData(name).RX(sigConfig, rxINNoisySig, ampThreshold);
        [Kp, SNRp] = calcOutputSNR(rxSig, txIFFTSig);
        [Ka, SNRa] = calcOutputSNR(rxDemapSig, txModSig);

        D_full   = rxFDSig - Ka .* txMapSig;          % txMapSig 在 null 上為 0
        distNull = sum(abs(D_full(sigConfig.nullIdx, :)) .^ 2, "all");
        distAll  = sum(abs(D_full) .^ 2, "all");
        beta = gather(distNull / distAll);

        allData(name).K_pooled(idxAmpThreshold) = Kp;
        allData(name).K_active(idxAmpThreshold) = Ka;
        allData(name).SNR_pooled(idxAmpThreshold) = SNRp;
        allData(name).SNR_active(idxAmpThreshold) = SNRa;
        allData(name).Beta(idxAmpThreshold) = beta;

    end

    % [dataMask, rxPGIRSig, rxPGIRFDSig, rxPGIRDemapSig] = rxPGIR(sigConfig, rxINNoisySig, ampThreshold, iterCount);
    % [K_PGIR_pooled(idxINprob, idxAmpThreshold), snr_PGIR_pooled(idxINprob, idxAmpThreshold)] = calcOutputSNR(rxPGIRSig, txIFFTSig);
    % [K_PGIR_active(idxINprob, idxAmpThreshold), snr_PGIR_active(idxINprob, idxAmpThreshold)] = calcOutputSNR(rxPGIRDemapSig, txModSig);

    % Kc = K_PGIR_active(idxINprob, idxAmpThreshold);
    % D_full   = rxPGIRFDSig - Kc .* txMapSig;          % txMapSig 在 null 上為 0
    % distNull = sum(abs(D_full(sigConfig.nullIdx, :)) .^ 2, "all");
    % distAll  = sum(abs(D_full) .^ 2, "all");
    % betaPGIR(idxINprob, idxAmpThreshold) = gather(distNull / distAll);

    % 順便記錄預算使用量
    % detCountPGIR(idxINprob, idxAmpThreshold) = gather(mean(sum(~dataMask, 1)));

    % rxEqSig  = rxPGIRDemapSig ./ K_active(idxINprob, idxAmpThreshold);
    % outDataBits  = demodulator(rxEqSig, modOrder);
    % errBitCount(idxAmpThreshold) = errBitCount(idxAmpThreshold) + sum(outDataBits(:) ~= inDataBits(:));

end


% % 自我檢查，應為 ~1e-12
% fprintf('beta identity err = %.3e\n', ...
%     max(abs(betaPGIR - (1 - snr_PGIR_pooled ./ snr_PGIR_active)), [], "all"));
% % 自我檢查，應為 ~1e-12
% fprintf('beta identity err = %.3e\n', ...
%     max(abs(betaReplaceClipBlank - (1 - snr_ReplaceClipBlank_pooled ./ snr_ReplaceClipBlank_active)), [], "all"));
% 
% % 這才是最終判準
% [~, iP] = max(snr_PGIR_pooled,  [], 2);
% [~, iA] = max(snr_PGIR_active,  [], 2);
% lin = @(i) sub2ind(size(snr_PGIR_active), (1:numel(INprobList))', i);
% penalty_dB = 10*log10(snr_PGIR_active(lin(iA)) ./ snr_PGIR_active(lin(iP)));
% disp(table(INprobList', ampThresholdList(iP)', ampThresholdList(iA)', penalty_dB, ...
%     VariableNames={'p','Topt_pooled','Topt_active','penalty_dB'}));

fprintf('Start calculate BER.\n');

fprintf('Start process %f probability at %s\n', INprob, getTimeStr);

allErrBitCount = dictionary;
for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    allErrBitCount(name) = struct('ErrorBitCount', gpuArray.zeros(1, length(ampThresholdList)));
end

% errBitCountPGIR = gpuArray.zeros(1, length(ampThresholdList));

for idxLoop = 1:(baseSigCount / sigBatchPerLoop)
    [rxINNoisySig, txIFFTSig, ~, ~, inDataBits] = makeFrame(sigConfig, sigBatchPerLoop, snr, INsnr, INprob);

    for idxAmpThreshold = 1:length(ampThresholdList)
        ampThreshold = ampThresholdList(idxAmpThreshold);

        for idxDeepMu = 1:length(deepMuList)
            deepMu = deepMuList(idxDeepMu);
            name = processName + string(deepMu);
            [~, ~, rxDemapSig] = allData(name).RX(sigConfig, rxINNoisySig, ampThreshold);
            Kc = allData(name).K_active(idxAmpThreshold);
            outDataBits = rxDemod(sigConfig, Kc, rxDemapSig);

            allErrBitCount(name).ErrorBitCount(idxAmpThreshold) = ...
                allErrBitCount(name).ErrorBitCount(idxAmpThreshold) + sum(outDataBits(:) ~= inDataBits(:));
        end

        % [~, ~, ~, rxPGIRDemapSig] = rxPGIR(sigConfig, rxINNoisySig, ampThreshold, iterCount);
        % Kc = K_PGIR_active(idxINprob, idxAmpThreshold);
        % rxEqSig = rxPGIRDemapSig ./ Kc;
        % rxPreDemodSig = reshape(rxEqSig, [sigConfig.numData*1 1*sigBatchPerLoop]);
        % outDataBitsPGIR = demodulator(rxPreDemodSig, sigConfig.modOrder, lower(sigConfig.modType));
        % errBitCountPGIR(idxAmpThreshold) = errBitCount(idxAmpThreshold) + sum(outDataBits(:) ~= inDataBits(:));

    end

    progress = idxLoop / (baseSigCount / sigBatchPerLoop);
    fprintf("Now done %.2f%% at %s\n", progress * 100, getTimeStr);
end

for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    allData(name).BER = allErrBitCount(name).ErrorBitCount / (baseSigCount * sigConfig.bitsPerOFDMSymbol);
end
% berPGIR(idxINprob, :) = errBitCountPGIR / (baseSigCount(idxINprob) * sigConfig.bitsPerOFDMSymbol);

fprintf('\n');
fprintf('%s\n', repmat('-', 1, 50));
fprintf('All process has done at %s\n', getTimeStr);

%%
res = struct();
res.scheme    = processName;
res.thr = ampThresholdList;
res.M = modOrder;
res.snrPooled = [];
res.snrActive = [];
res.ber = [];
for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    res.snrPooled = [res.snrPooled; allData(name).SNR_pooled];
    res.snrActive = [res.snrActive; allData(name).SNR_active];
    res.ber = [res.ber; allData(name).BER];
end
res.p = INprob;

% tbl = optTable(res);
% save(sprintf('optSummary_M%d_p%g.mat', modOrder, INprob), 'tbl');
% disp(tbl);

%% plot figure
linewidth = 1.5;
markersize = 10;
markerSpace = 5;
markerIdx = markerSpace:markerSpace:length(ampThresholdList);
c = orderedcolors('gem12');
lenC = size(c, 2);
markerList = {'none', 'o', '+', 'x', '*', 's', 'd'};
styles = {'-', '--', '-.', '-o', '-+', '-x', '-*'};

% figure 1
figure;
hold on; grid on;  box on;

for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    plot(ampThresholdList, 10 * log10(allData(name).SNR_pooled), '-', Color=c(mod(idxDeepMu-1, lenC)+1, :), ...
        DisplayName=sprintf('SNR^{%s}_pooled', name), LineWidth=linewidth, ...
        Marker=markerList{1}, MarkerIndices=markerIdx, MarkerSize=markersize);
    plot(ampThresholdList, 10 * log10(allData(name).SNR_active), '--', Color=c(mod(idxDeepMu-1, lenC)+1, :), ...
        DisplayName=sprintf('SNR^{%s}_active', name), LineWidth=linewidth, ...
        Marker=markerList{1}, MarkerIndices=markerIdx, MarkerSize=markersize);
end
% plot(ampThresholdList, 10 * log10(snr_PGIR_pooled(idxINprob, :)), '-', Color=c(idxINprob, :), ...
%     DisplayName='snr_{pooled}', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);
% plot(ampThresholdList, 10 * log10(snr_PGIR_active(idxINprob, :)), '--', Color=c(idxINprob, :), ...
%     DisplayName='snr_{active}', MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);


nColor = length(deepMuList);
nStyle = 3;

colorTexts = cellstr('\mu='+string(deepMuList));
% styleTexts = {"snr_{pooled}", "snr_{active}", "snr^{blanking}_{pooled}", "snr^{blanking}_{active}"};
styleTexts = {'SNR_{pooled}', 'SNR_{active}', processName};
styleLines = {'-', '--', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
styleMarkers = {'none', 'none', markerList{:}};

hColors = cell(nColor, 1);
hStyles = cell(nStyle, 1);

for idx = 1:nColor
    hColors{idx} = plot(nan, nan, LineStyle="-", Marker="none", LineWidth=1.5, ...
        Color=c(mod(idx-1,lenC)+1,:), DisplayName=colorTexts{idx});
end

for idx = 1:nStyle
    hStyles{idx} = plot(nan, nan, Color="k", LineWidth=1.5, ...
        LineStyle=styleLines{idx}, Marker=styleMarkers{idx}, MarkerSize=8, ...
        DisplayName=styleTexts{idx});
end

hold off;
legend([hColors{:}, hStyles{:}]);
ymax = ceil(max(10 * log10(res.snrActive), [], 'all') / 5) * 5;
ylim([-25 ymax]);
xlabel('Unknown Data Threshold (T)');
ylabel('Output SNR (\gamma), dB');
title(sprintf('N = %d, Oversampling %d, %d QAM, p = %s, SNR comparison', ...
    fftSize, OSFactor, modOrder, string(INprob)), Interpreter="latex", FontSize=16);

% figure 2
figure;
hold on; grid on;  box on;

for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    % plot(ampThresholdList, allData(name).Beta(idxINprob, :)  / (1 - 1 / OSFactor), styles{idxProcess}, ...
    %     Color=c(idxINprob, :), DisplayName=sprintf('p=%.3f, %s', INprobList(idxINprob), name), ...
    %     LineWidth=linewidth, MarkerIndices=markerIdx, MarkerSize=markersize);
    plot(ampThresholdList, allData(name).Beta, styles{1}, ...
        Color=c(mod(idxDeepMu-1,lenC)+1, :), DisplayName=sprintf('%s', '\mu=' + string(deepMu)), ...
        LineWidth=linewidth, MarkerIndices=markerIdx, MarkerSize=markersize);
end
% plot(ampThresholdList, betaPGIR(idxINprob,:) / (1 - 1 / OSFactor), '-', Color=c(idxINprob, :), ...
%     DisplayName=sprintf("p=%.3f", INprobList(idxINprob)), MarkerIndices=markerIdx, ...
%     LineWidth=linewidth, MarkerSize=markersize);

plot(ampThresholdList, ones(length(ampThresholdList), 1), ':', Color="k", ...
    DisplayName="\beta=1", MarkerIndices=markerIdx, LineWidth=linewidth, MarkerSize=markersize);

hold off;
legend;
xlabel('Unknown Data Threshold (T)');
ylabel('\beta');
title(sprintf('N = %d, Oversampling %d, p = %s, $\\beta$', fftSize, OSFactor, string(INprob)), ...
    Interpreter="latex", FontSize=16);

% figure 3
figure;
hold on; grid on;  box on;

for idxDeepMu = 1:length(deepMuList)
    deepMu = deepMuList(idxDeepMu);
    name = processName + string(deepMu);
    plot(ampThresholdList, allData(name).BER, styles{1}, ...
        Color=c(mod(idxDeepMu-1,lenC)+1, :), DisplayName=sprintf('%s', '\mu=' + string(deepMu)), ...
        LineWidth=linewidth, MarkerIndices=markerIdx, MarkerSize=markersize);
end
% semilogy(ampThresholdList, berPGIR(idxINprob, :), '-', Color=c(idxINprob, :), ...
%     DisplayName=sprintf("p=%.3f", INprobList(idxINprob)), MarkerIndices=markerIdx, ...
%     LineWidth=linewidth, MarkerSize=markersize);

hold off;
set(gca, 'YScale', 'log')
legend;
xlabel('Unknown Data Threshold (T)');
ylabel('BER');
title(sprintf('N = %d, Oversampling %d, %d QAM, %s, BER', fftSize, OSFactor, modOrder, string(INprob)), ...
    Interpreter="latex", FontSize=16);

% figure 4
% figure
% activeMask = zeros(fftSize, 1);
% activeMask(dataIdx) = 1;
% hold on; grid on;  box on;
% 
% for idxDeepMu = 1:length(deepMuList)
%     deepMu = deepMuList(idxDeepMu);
%     name = processName + string(deepMu);
% 
%     snrPooled = allData(name).SNR_pooled;
%     snrActive = allData(name).SNR_active;
%     betaResult = betaMemorylessTheory(ampThresholdList, INprob, Scheme='deepclip', ...
%             N=fftSize, Los=OSFactor, SNRdB=snr, SINRdB=INsnr, ActiveMask=activeMask, TbFactor=1.4, Mu=deepMu);
%     snrCal = snrPooled ./ (1 - betaResult.beta.');
%     plot(ampThresholdList, 10 * log10(snrPooled), '-', ...
%         Color=c(mod(idxDeepMu-1,lenC)+1, :), DisplayName=sprintf('\\mu=%s, pooled', string(deepMu)), ...
%         LineWidth=linewidth, MarkerIndices=markerIdx, MarkerSize=markersize);
%     plot(ampThresholdList, 10 * log10(snrActive), '--x', ...
%         Color=c(mod(idxDeepMu-1,lenC)+1, :), DisplayName=sprintf('\\mu=%s, active', string(deepMu)), ...
%         LineWidth=linewidth, MarkerIndices=markerIdx, MarkerSize=markersize);
%     plot(ampThresholdList, 10 * log10(snrCal), '-.o', ...
%         Color=c(mod(idxDeepMu-1,lenC)+1, :), DisplayName=sprintf('\\mu=%s, pooled to active', string(deepMu)), ...
%         LineWidth=linewidth, MarkerIndices=markerIdx, MarkerSize=markersize);
% end
% % semilogy(ampThresholdList, berPGIR(idxINprob, :), '-', Color=c(idxINprob, :), ...
% %     DisplayName=sprintf("p=%.3f", INprobList(idxINprob)), MarkerIndices=markerIdx, ...
% %     LineWidth=linewidth, MarkerSize=markersize);
% 
% title('Deep Clipping SNR')
% legend
% xlabel('Unknown Data Threshold (T)');
% ylabel('Output SNR (\gamma), dB');

%%
function [structure] = constructData(size, rxHandle)
    structure = struct('K_pooled', zeros(size), ...
            'K_active', zeros(size), ...
            'SNR_pooled', zeros(size), ...
            'SNR_active', zeros(size), ...
            'Beta', zeros(size), ...
            'BER', zeros(size), ...
            'RX', rxHandle);
end

function [K_0, outSNR] = calcOutputSNR(inSig, refSig)
    K_0 = sum(inSig .* conj(refSig), "all") / sum(abs(refSig) .^ 2, "all");
    sigPower = abs(K_0) ^ 2 * sum(abs(refSig) .^ 2, "all");
    errPower = sum(abs(inSig - K_0 .* refSig) .^ 2, "all");

    outSNR = sigPower / errPower;
end

function [rxINNoisySig, txIFFTSig, txMapSig, txModSig, inDataBits] = makeFrame(cfg, sigCount, snr, INsnr, INprob)
    % Tx
    inDataBits = randomBits([cfg.bitsPerOFDMSymbol*1 1*sigCount], OutputLocation='gpu');
    txModSig = modulator(inDataBits, cfg.modOrder, lower(cfg.modType));
    txModSig = reshape(txModSig, [cfg.numData 1 1 sigCount]);
    txMapSig = scMap(txModSig, cfg.fftSize, cfg.nullIdx);
    txIFFTSig = sqrt(cfg.OSFactor) .* sqrt(cfg.fftSize) .* ifft(txMapSig, cfg.fftSize, 1);
    txSig = reshape(txIFFTSig, [cfg.fftSize*1 1 sigCount]);

    % Noise
    noisePower = cfg.sigPowerRef / (10 ^ (snr / 10));
    noise = awgnx(size(txSig), noisePower, txSig(1));
    rxNoisySig = txSig + noise;

    % Impulsive Noise
    INPower = cfg.sigPowerRef / (10 ^ (INsnr / 10));
    [impulsiveNoise, ~] = IN(size(rxNoisySig), INPower, INprob, rxNoisySig(1));
    rxINNoisySig = rxNoisySig + impulsiveNoise;

end

function [dataMask, rxPGIRSig, rxPGIRFDSig, rxPGIRDemapSig] = rxPGIR(cfg, rxINNoisySig, threshold, iterCount)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    rxINSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);

    dataMask = abs(rxINSig) < threshold;
    rxPGIRSig = PGIR(rxINSig, cfg.refSig, dataMask, cfg.transMask, iterCount);
    rxPGIRFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxPGIRSig, cfg.fftSize, 1);
    rxPGIRDemapSig = scDemap(rxPGIRFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxPGIRSig, rxPGIRFDSig, rxPGIRDemapSig] = rxPGIR1(cfg, rxINNoisySig, threshold, iterCount)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    rxINSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);

    dataMask = abs(rxINSig) < threshold;
    rxPGIRSig = PGIR(rxINSig, cfg.refSig, dataMask, cfg.transMask, iterCount);
    rxPGIRFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxPGIRSig, cfg.fftSize, 1);
    rxPGIRDemapSig = scDemap(rxPGIRFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxBlankSig, rxBlankFDSig, rxBlankDemapSig] = rxBlank(cfg, rxINNoisySig, threshold)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    tempSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);
    rxBlankSig = blanking(tempSig, threshold);
    rxBlankFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxBlankSig, cfg.fftSize, 1);
    rxBlankDemapSig = scDemap(rxBlankFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxClipSig, rxClipFDSig, rxClipDemapSig] = rxClip(cfg, rxINNoisySig, threshold)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    tempSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);
    rxClipSig = clipping(tempSig, threshold);
    rxClipFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxClipSig, cfg.fftSize, 1);
    rxClipDemapSig = scDemap(rxClipFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxClipBlankSig, rxClipBlankFDSig, rxClipBlankDemapSig] = rxClipBlank(cfg, rxINNoisySig, threshold, tClipBlank)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    tempSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);
    rxClipBlankSig = clipBlank(tempSig, threshold, tClipBlank(threshold));
    rxClipBlankFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxClipBlankSig, cfg.fftSize, 1);
    rxClipBlankDemapSig = scDemap(rxClipBlankFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxDeepClipSig, rxDeepClipFDSig, rxDeepClipDemapSig] = rxDeepClip(cfg, rxINNoisySig, threshold, deepMu)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    tempSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);
    rxDeepClipSig = deepClip(tempSig, threshold, deepMu);
    rxDeepClipFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxDeepClipSig, cfg.fftSize, 1);
    rxDeepClipDemapSig = scDemap(rxDeepClipFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxReplaceSig, rxReplaceFDSig, rxReplaceDemapSig] = rxReplace(cfg, rxINNoisySig, threshold, sigPower)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    tempSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);
    rxReplaceSig = replacement(tempSig, threshold, sigPower);
    rxReplaceFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxReplaceSig, cfg.fftSize, 1);
    rxReplaceDemapSig = scDemap(rxReplaceFDSig, cfg.fftSize, cfg.nullIdx);
end

function [rxClipReplaceBlankSig, rxClipReplaceBlankFDSig, rxClipReplaceBlankDemapSig] = rxClipReplaceBlank(cfg, rxINNoisySig, threshold, sigPower, tClipReplace, tClReBlank)
    % Rx
    sigCount = size(rxINNoisySig, 3);
    tempSig = reshape(rxINNoisySig, [cfg.fftSize 1 1 sigCount]);
    rxClipReplaceBlankSig = clipReplaceBlank(tempSig, threshold, tClipReplace(threshold), ...
                                tClReBlank(threshold), sigPower);
    rxClipReplaceBlankFDSig = 1 / (sqrt(cfg.OSFactor) * sqrt(cfg.fftSize)) * fft(rxClipReplaceBlankSig, cfg.fftSize, 1);
    rxClipReplaceBlankDemapSig = scDemap(rxClipReplaceBlankFDSig, cfg.fftSize, cfg.nullIdx);
end

function [outDataBits] = rxDemod(config, Kc, rxSig)
    sigCount = size(rxSig, 4);

    rxEqSig = rxSig ./ Kc;
    rxPreDemodSig = reshape(rxEqSig, [config.numData*1 1*sigCount]);
    outDataBits = demodulator(rxPreDemodSig, config.modOrder, lower(config.modType));
end
