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
sigBatchPerLoop = 10000;        % Every loop test signals
ebn0List = 10:1:20;             % Energy per bit to noise power spectral density ratio(dB)

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprob = 0.01;                  % Probability of impulsive noise occurring

% Non-linear process paramter
% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

tClipBlank = @(T) (T * 1.4);    % Clipping-Blanking
deepMu = 0.5;                   % Deep-Clipping
tClipReplace = @(T) (T * 1.2);  % Clipping-Replacement-Blanking Clipping
tClReBlank = @(T) (T * 1.4);    % Clipping-Replacement-Blanking Blanking

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
transMask = zeros(fftSize, 1);
transMask([nullIdx; pilotIdx]) = 1;
refSigFD = zeros(fftSize, 1, 1, sigBatchPerLoop);
refSigFD(pilotIdx, :, :, :) = pilotVal;
refSig = sqrt(fftSize) * ifft(refSigFD, fftSize, 1);

sigConfig = struct('numData', numData, 'modOrder', modOrder, 'modType', modType, ...
    'fftSize', fftSize, 'nullIdx', nullIdx, 'pilotIdx', pilotIdx, 'transMask', transMask, ...
    'refSig', refSig, 'sigPowerRef', sigPowerRef, 'bitsPerOFDMSymbol', bitsPerOFDMSymbol);

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
fprintf('Start process %f probability at %s\n', INprob, getTimeStr);

for idxEbn0 = 1:size(ebn0List, 2)
    % SNR calculation
    snr = ebn0List(idxEbn0) + 10 * log10(bitsPerModSymbol) ...
        + 10 * log10(numData / fftSize);
    % Calculate the amount of test signals based on SNR
    totalSigCount = (10 ^ floor(snr / 10)) * baseSigCount;
    fprintf('EbN0 = %2d, max signal number = %d\n', ebn0List(idxEbn0), totalSigCount);
    fprintf('Start at %s\n', getTimeStr);

    % Calculate noise power & impulsive noise power
    noisePower = sigPowerRef / (10 ^ (snr / 10));
    INPower = sigPowerRef / (10 ^ (INsnr / 10));
    % BER storage depends on EbN0
    tempBERPGIR = zeros(1, totalSigCount / sigBatchPerLoop);
    tempBERBlank = zeros(1, totalSigCount / sigBatchPerLoop);
    tempBERClip = zeros(1, totalSigCount / sigBatchPerLoop);
    tempBERClipBlank = zeros(1, totalSigCount / sigBatchPerLoop);
    % tempBERDeepClip = zeros(1, totalSigCount / sigBatchPerLoop);
    % tempBERReplace = zeros(1, totalSigCount / sigBatchPerLoop);
    % tempBERReplaceClipBlank = zeros(1, totalSigCount / sigBatchPerLoop);

    optThresPGIR = getOptThresPGIR(sigPowerRef, noisePower, INPower, INprob, ...
            fftSize, nullIdx, 'nIter', iterCount, 'M', 1000);
    optThresBlank = getOptThresBlanking(sigPowerRef, noisePower, INPower, INprob);
    optThresClip = getOptThresClipping(sigPowerRef, noisePower, INPower, INprob);
    optThresClipBlank = getOptThresClipBlanking(sigPowerRef, noisePower, INPower, INprob);

    parfor idxRun = 1:(totalSigCount / sigBatchPerLoop)
        % Tx
        inDataBits = randomBits([bitsPerOFDMSymbol*1 1*sigBatchPerLoop]);
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

        rxINSig = reshape(rxINNoisySig, [fftSize 1 1 sigBatchPerLoop]);

        ampThreshold = optThresPGIR;
        dataMask = abs(rxINSig) < ampThreshold;
        rxPGIRSig = PGIR(rxINSig, refSig, dataMask, transMask, iterCount);
        outPGIRDataBits = rxDemod(sigConfig, rxPGIRSig);

        % ampThreshold = optTList(2);
        ampThreshold = optThresBlank;
        rxBlankSig = blanking(rxINSig, ampThreshold);
        outBlankDataBits = rxDemod(sigConfig, rxBlankSig);

        ampThreshold = optThresClip;
        rxClipSig = clipping(rxINSig, ampThreshold);
        outClipDataBits = rxDemod(sigConfig, rxClipSig);

        ampThreshold = optThresClipBlank;
        rxClipBlankSig = clipBlank(rxINSig, ampThreshold, tClipBlank(ampThreshold));
        outClipBlankDataBits = rxDemod(sigConfig, rxClipBlankSig);

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

fprintf('\n');
fprintf('%s\n', repmat('-', 1, 50));
fprintf('All process has done at %s\n', getTimeStr);

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

%%
function [outDataBits] = rxDemod(config, rxSig)
    sigCount = size(rxSig, 4);

    rxFDSig = 1 / sqrt(config.fftSize) .* fft(rxSig, config.fftSize, 1);
    rxDemapSig = scDemap(rxFDSig, config.fftSize, config.nullIdx, config.pilotIdx);
    rxPreDemodSig = reshape(rxDemapSig, [config.numData*1 1*sigCount]);
    outDataBits = demodulator(rxPreDemodSig, config.modOrder, lower(config.modType));
end
