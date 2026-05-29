clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
fftSize = 256;                   % FFT size
dataFactor = [15 1]; % data, null ratio
nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2)); % Null Subcarrier Index
cpLen = fftSize * 1 / 4;        % Cyclic Prefix size
channelLen = 8;                 % Multipath length in rayleight distribution (no LoS)
kFactor = 8;
modOrder = 16;                  % The point amount of constellation
modType = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')
nTX = 1;
nRX = 1;
baseSigCount = 10000;           % testing signal numbers (will multiply a factor)
sigPerLoop = 10000;               % Every loop test signals
ebn0 = 30;              % Energy per bit to noise power spectral density ratio(dB)
inCount = 2;
INsnr = -15;
INprob = 0.01;
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
cpAdder = @(sig, len) sig([end-len+1:end, 1:end], :, :);
cpRemover = @(sig, len) sig(len+1:end, :, :);

transMask = gpuArray.zeros(fftSize, 1);
transMask(nullIdx) = 1;
transMask = repmat(transMask, 1, sigPerLoop);
sigRef = gpuArray.zeros(fftSize, sigPerLoop);

%% data storage
nmseList = cell(1, length(iterCountList));
avgNMSEList = zeros(1, length(iterCountList));
berINPGIRList = zeros(1, length(iterCountList));

%% CP-OFDM

% SNR calculation
snr = ebn0 + 10 * log10(bitsPerModSymbol) ...
    + 10 * log10(numData / (fftSize + cpLen));

fprintf('EbN0 = %2d, max signal number = %d\n', ebn0, baseSigCount);

tempBERINPGIR = zeros(length(iterCountList), baseSigCount / sigPerLoop);

simProcess = 0;

fprintf("Start simulation at %s...\n", datetime('now', TimeZone='local', Format='MM-dd HH:mm:ss'));

for idxRun = 1:(baseSigCount / sigPerLoop)

    % Tx
    inDataBits = randomBits([bitsPerOFDMSymbol sigPerLoop nTX]);
    txModSig = modulator(inDataBits, modOrder);
    txMapSig = scMap(txModSig, fftSize, nullIdx);
    txIFFTSig = sqrt(fftSize) .* ifft(txMapSig, fftSize, 1);
    txOFDMSig = cpAdder(txIFFTSig, cpLen);
    
    % Channel
    sigPower = calcPower(txOFDMSig);
    % [fadedSig, channel] = ricianChannel(txOFDMSig, fftSize, channelLen, kFactor, nRX);
    fadedSig = txOFDMSig;
    channel = 1;
    
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
    
    % Orig
    rxNoCPSig = cpRemover(rxNoisySig, cpLen);
    rxFFTSig = 1 / sqrt(fftSize) .* fft(rxNoCPSig, fftSize, 1);
    rxEQSig = equalizer(rxFFTSig, channel);
    rxDemapSig = scDemap(rxEQSig, fftSize, nullIdx);
    outDataBits = demodulator(rxDemapSig, modOrder);
    % [~, tempBER(idxRun)] = biterr(inDataBits(:), outDataBits(:));
    
    % Orig + IN
    rxINNoCPSig = cpRemover(rxINNoisySig, cpLen);
    rxINFFTSig = 1 / sqrt(fftSize) .* fft(rxINNoCPSig, fftSize, 1);
    rxINEQSig = equalizer(rxINFFTSig, channel);
    rxINDemapSig = scDemap(rxINEQSig, fftSize, nullIdx);
    outINDataBits = demodulator(rxINDemapSig, modOrder);
    % [~, tempBERIN(idxRun)] = biterr(inDataBits(:), outINDataBits(:));
    
    % Orig + IN with PGIR
    % rxIndex = sub2ind(size(rxINFFTSig), orgIndex, repmat(1:sigPerLoop, inCount, 1));
    % dataMask = ones(fftSize, sigPerLoop);
    % dataMask(rxIndex) = 0;
    dataMask = gpuArray(~happenIdx((cpLen+1):end, :,:,:));
    rxPGIRTDSig = sqrt(fftSize) .* ifft(rxINEQSig, fftSize, 1);
    
    for idxIterCount = 1:length(iterCountList)
        iterCount = iterCountList(idxIterCount);
    
        rxPGIRSig = PGIR(rxPGIRTDSig, sigRef, dataMask, transMask, iterCount);
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

%%

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