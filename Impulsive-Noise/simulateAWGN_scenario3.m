clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
fftSize = 256;                  % FFT size
dataFactor = [7 1];             % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')

% Simulation parameter
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigBatchPerLoop = 10000;             % Every loop test signals
ebn0 = 30;                      % Energy per bit to noise power spectral density ratio(dB)

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprob = 0.01;                  % Probability of impulsive noise occurring

% PGIR parameter
% iterCountList = [1 10 20 50 100];
% iterCountList = [200 500 1000 2000];
% iterCountList = [1 10 20 50 100 200 500 1000 2000];
iterCountList = [1:10:100 100];

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
nmseList = cell(1, length(iterCountList));
avgNMSEList = zeros(1, length(iterCountList));
berINPGIRList = zeros(1, length(iterCountList));

%% CP-OFDM

% SNR calculation
snr = ebn0 + 10 * log10(bitsPerModSymbol) ...
    + 10 * log10(numData / fftSize);

fprintf('EbN0 = %2d, max signal number = %d\n', ebn0, baseSigCount);
fprintf('Start at %s\n', getTimeStr);

% Calculate noise power & impulsive noise power
noisePower = sigPowerRef / (10 ^ (snr / 10));
INPower = sigPowerRef / (10 ^ (INsnr / 10));
% BER storage depends on EbN0
tempBERINPGIR = zeros(length(iterCountList), baseSigCount / sigBatchPerLoop);

simProcess = 0;

for idxRun = 1:(baseSigCount / sigBatchPerLoop)
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
    % Orig
    rxSig = reshape(rxNoisySig, [fftSize 1 1 sigBatchPerLoop]);
    rxFFTSig = 1 / sqrt(fftSize) .* fft(rxSig, fftSize, 1);
    rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);

    % Orig + IN
    rxINSig = reshape(rxINNoisySig, [fftSize 1 1 sigBatchPerLoop]);
    rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINSig, fftSize, 1);
    rxINDemapSig = scDemap(rxINFFTSig, fftSize, nullIdx);

    % Orig + IN with PGIR
    dataMask = gpuArray(~reshape(happenIdx, [fftSize 1 1 sigBatchPerLoop]));

    for idxIterCount = 1:length(iterCountList)
        iterCount = iterCountList(idxIterCount);

        rxPGIRSig = PGIR(rxINSig, refSig, dataMask, transMask, iterCount);
        rxPGIRFDSig = 1 / sqrt(fftSize) .* fft(rxPGIRSig, fftSize, 1);
        rxPGIRDemapSig = scDemap(rxPGIRFDSig, fftSize, nullIdx);
        rxPGIRPreDemodSig = reshape(rxPGIRDemapSig, [numData*1 1*sigBatchPerLoop]);
        outINPGIRDataBits = demodulator(rxPGIRPreDemodSig, modOrder, lower(modType));
        nmse = sum(abs(rxPGIRSig - txIFFTSig) .^ 2, 1) ./ sum(abs(txIFFTSig) .^ 2, 1);

        nmseList{idxIterCount} = [nmseList{idxIterCount} nmse];
        [~, tempBERINPGIR(idxIterCount, idxRun)] = biterr(inDataBits(:), outINPGIRDataBits(:));
    
    end

    if mod(idxRun, floor(length(1:(baseSigCount / sigBatchPerLoop)) / 10)) == 0
        simProcess = simProcess + 1;
        fprintf('Process has done %3d%% at %s\n', simProcess * 10, ...
            datetime('now', TimeZone='local', Format='MM-dd HH:mm:ss'));
    end

end

berINPGIRList(:) = mean(tempBERINPGIR, 2);
for idxIterCount = 1:length(iterCountList)
    avgNMSEList(idxIterCount) = mean(nmseList{idxIterCount});
end

%% plot figure
c = orderedcolors('gem12');
x_plot = iterCountList;

figure;
hold on;
yyaxis left;
idxC = 1;
semilogx(x_plot, avgNMSEList, Color=c(idxC,:), LineStyle='-', Marker='none', ...
    DisplayName="$\overline{\mathrm{NMSE}}$");

ylabel('$\overline{\mathrm{NMSE}}$', 'Interpreter','latex', 'FontSize', 14)
% hold off;
% legend('Interpreter', "latex")
% xlabel('Iteration Count(s) ($i_t$)', 'Interpreter','latex', 'FontSize', 14);
% grid on;

% figure
% hold on;
yyaxis right;
idxC = 1;
semilogx(x_plot, berINPGIRList, Color=c(idxC,:), LineStyle='--', Marker='none', ...
    DisplayName="BER");

ylabel('BER');

hold off;

% xscale log;
legend('Interpreter', "latex")
xlabel('Iteration Count(s) ($i_t$)', 'Interpreter','latex', 'FontSize', 14);
grid on;

return
%%
figure
hold on;
for i = 1:length(iterCountList)
    barHistogram(nmseList{i}, 0.0001, DisplayName=sprintf("iteration %d time(s)", iterCountList(i)));
end
hold off;
legend
colororder('gem12');
