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
sigPerLoop = 10000;             % Every loop test signals
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
nmseList = cell(1, length(iterCountList));
avgNMSEList = zeros(1, length(iterCountList));
berINPGIRList = zeros(1, length(iterCountList));

%% CP-OFDM

% SNR calculation
snr = ebn0 + 10 * log10(bitsPerModSymbol) ...
    + 10 * log10(numData / fftSize);

fprintf('EbN0 = %2d, max signal number = %d\n', ebn0, baseSigCount);

tempBERINPGIR = zeros(length(iterCountList), baseSigCount / sigPerLoop);

simProcess = 0;

fprintf("Start simulation at %s...\n", datetime('now', TimeZone='local', Format='MM-dd HH:mm:ss'));

for idxRun = 1:(baseSigCount / sigPerLoop)
    % Tx
    inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop]);
    txModSig = modulator(inDataBits, modOrder);
    txMapSig = scMap(txModSig, fftSize, nullIdx);
    txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);

    sigPower = calcPower(txIFFTSig);

    % Noise
    [noise, noisePower] = awgnx(size(txIFFTSig), snr, sigPower, txIFFTSig(1));
    rxNoisySig = txIFFTSig + noise;

    % Impulsive Noise
    [impulsiveNoise, ~, happenIdx] = IN(size(rxNoisySig), INsnr, INprob, sigPower, rxNoisySig(1));
    rxINNoisySig = rxNoisySig + impulsiveNoise;

    % Rx
    % Orig
    rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoisySig, fftSize, 1);
    rxDemapSig = scDemap(rxFFTSig, fftSize, nullIdx);

    % Orig + IN
    rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINNoisySig, fftSize, 1);
    rxINDemapSig = scDemap(rxINFFTSig, fftSize, nullIdx);

    % Orig + IN with PGIR
    dataMask = gpuArray(~happenIdx);

    for idxIterCount = 1:length(iterCountList)
        iterCount = iterCountList(idxIterCount);

        rxPGIRSig = PGIR(rxINNoisySig, txIFFTSig, dataMask, transMask, iterCount);
        rxPGIRFDSig = 1 / sqrt(fftSize) .* fft(rxPGIRSig, fftSize, 1);
        rxPGIRDemapSig = scDemap(rxPGIRFDSig, fftSize, nullIdx);
        outINPGIRDataBits = demodulator(rxPGIRDemapSig, modOrder);
        nmse = sum(abs(rxPGIRSig - txIFFTSig) .^ 2, 1) ./ sum(abs(txIFFTSig) .^ 2, 1);

        nmseList{idxIterCount} = [nmseList{idxIterCount} nmse];
        [~, tempBERINPGIR(idxIterCount, idxRun)] = biterr(inDataBits(:), outINPGIRDataBits(:));
    
    end

    if mod(idxRun, floor(length(1:(baseSigCount / sigPerLoop)) / 10)) == 0
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
