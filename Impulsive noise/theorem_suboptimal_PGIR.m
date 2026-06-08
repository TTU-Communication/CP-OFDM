clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%%
fftSize = 256;
% dataFactor = [7 1]; % data, null ratio
% nullIdx = getNullIdx(fftSize, fftSize / sum(dataFactor) * dataFactor(2));
nullIdx = getNullIdx(fftSize);
[pilotIdx, pilotVal] = getPilotIdxAndVal(fftSize);
iterCount = 50;
snr = 25;
sinr = -15;
INHappenPList = [0.001 0.01 0.1];
TRange = linspace(0.1, 15, 150);

Msample = 3000;

%%
powerS = (fftSize - length(nullIdx)) / fftSize;
powerW = powerS / 10 ^ (snr / 10);
powerG = powerS / 10 ^ (sinr / 10);

powerList = powerS + [powerW; powerW + powerG];

gammaBlank = nan(length(TRange), length(INHappenPList));
gammaPGIR  = nan(length(TRange), length(INHappenPList));   % closed-form + K & E corrections

fprintf('Running  (%d T points, %d MC samples each) ...\n', ...
        length(TRange), Msample);

for idxINp = 1:length(INHappenPList)
    INp = INHappenPList(idxINp);

    pList = [1 - INp; INp];

    t0 = tic;

    parfor TIdx = 1:length(TRange)
        T = TRange(TIdx);
        expList = exp(-T .^ 2 ./ powerList);

        KBlank = 1 - sum(pList .* (1 + T .^ 2 ./ powerList) .* expList, 1);
        EoutBlank = sum(pList .* (powerList - (T .^ 2 + powerList) .* expList), 1);
        gammaBlank(TIdx, idxINp) = (EoutBlank / (powerS * abs(KBlank) ^ 2) - 1) ^ -1;

        KPGIRSig = sum(pList .* powerS .* (1 + powerS .* T .^ 2 ./ powerList .^ 2) .* expList, 1);
        EoutPGIRSig = KPGIRSig;

        U = sum(pList .* expList, 1);
        [powerE, errorSigCon] = reconErr(fftSize, nullIdx, T, INp, ...
                                 powerS, powerW, powerG, ...
                                 iterCount, Msample, pilotIdx, pilotVal);
        KPGIR = KBlank + (KPGIRSig + errorSigCon * U) / powerS;
        EoutPGIR = EoutBlank + EoutPGIRSig + (powerE + 2 * real(errorSigCon)) * U;

        gammaPGIR(TIdx, idxINp) = (EoutPGIR / (powerS * abs(KPGIR) ^ 2) - 1) ^ -1;
    end
    fprintf('   p = %-6g  done in %5.1f s\n', INp, toc(t0));

end

%%
colors = orderedcolors('gem12');
linewidth = 1.5;
figure; hold on; grid on; box on;
for idxINp = 1:length(INHappenPList)
        plot(TRange, snr_db(gammaPGIR(:, idxINp)), '-',  Color=colors(idxINp,:), ...
        DisplayName=sprintf('p = %g', INHappenPList(idxINp)), LineWidth=linewidth);
    plot(TRange, snr_db(gammaBlank(:, idxINp)), '--', Color=colors(idxINp,:), ...
        LineWidth=linewidth);
end
hold off;
xlabel('Unknown Data Thershold (T)');
ylabel('Output SNR (\gamma), dB');
ylim([-25 25]);
legend

%%
function db = snr_db(num)
    idx = num > 0;
    db(idx) = 10 * log10(num(idx));
    db(~idx) = nan;
end

function [powerE, errorSigCon] = reconErr(fftSize, nullMask, T, p, powerS, powerW, powerG, iterCount, Msample, pilotIdx, pilotVal)

    if nargin < 10, pilotIdx = []; end
    if nargin < 11, pilotVal = []; end
    if isempty(pilotIdx) || isempty(pilotVal)
        pilotIdx = [];
        pilotVal = [];
    end
    pilotIdx    = pilotIdx(:);
    pilotVal    = pilotVal(:);

    totabssquE = 0;
    toterrorSig = 0;
    cnt     = 0;

    for m = 1:Msample
        % ---- OFDM signal (with null subcarriers) ----------------------
        X = (randn(fftSize,1) + 1j*randn(fftSize,1)) / sqrt(2);
        X(nullMask) = 0;
        X(pilotIdx) = pilotVal;             % insert the known pilot symbols
        x = sqrt(fftSize) .* ifft(X);

        % ---- Bernoulli-Gaussian noise --------------------------------
        w = (randn(fftSize,1) + 1j*randn(fftSize,1)) / sqrt(2) * sqrt(powerW);
        g = (randn(fftSize,1) + 1j*randn(fftSize,1)) / sqrt(2) * sqrt(powerG);
        b = rand(fftSize,1) < p;
        u = w + b.*g;
        r = x + u;

        unk = abs(r) > T;
        kno = ~unk;
        if ~any(unk),  continue;  end

        % ---- PGIR  xhat^(i+1) = P_D(P_T(xhat^(i))),  xhat^(0) = 0 ----
        xhat = zeros(fftSize,1);
        for i = 0:iterCount
            Xhat = 1 / sqrt(fftSize) .* fft(xhat);
            Xhat(nullMask) = 0;             % null  subcarriers -> 0
            Xhat(pilotIdx) = pilotVal;   % pilot subcarriers -> known value
            xhat = sqrt(fftSize) .* ifft(Xhat);
            xhat(kno) = r(kno);             % trusted (un-blanked) samples
        end

        error = xhat(unk) - x(unk);
        totabssquE = totabssquE + sum(abs(error).^2);
        toterrorSig = toterrorSig + sum(error .* conj(x(unk)));
        cnt     = cnt + sum(unk);
    end

    if cnt == 0
        powerE = 0;  errorSigCon = 0;
    else
        powerE      = totabssquE / cnt;
        errorSigCon = toterrorSig / cnt;
    end

end