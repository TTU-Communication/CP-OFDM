clc; clear;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core')));

%% parameter setting
% Signal parameter
numData = 64;                   % Data subcarrier size
OSFactor = 4;                   % oversampling factor

snr = 25;                       % Noise-to-Signal ratio

% Impulsive Noise parameter
INsnr = -15;                    % Impulsive-Noise-to-Signal ratio
INprobList = [0.001 0.01 0.1];  % Probability of impulsive noise occurring

% Non-linear process paramter
ampThresholdList = 0.1:0.1:15;  % Amplitude threshold

% PGIR parameter
iterCount = 20;                 % Number of PGIR iterations

Msample = 3000;

%%
fftSize = numData * OSFactor;
nullIdx = getNullIdx(fftSize, (OSFactor - 1) * numData);
powerS = (fftSize - length(nullIdx)) / fftSize * OSFactor;
powerW = powerS / 10 ^ (snr / 10);
powerG = powerS / 10 ^ (INsnr / 10);

powerList = powerS + [powerW; powerW + powerG];

gammaBlank = nan(length(ampThresholdList), length(INprobList));
gammaPGIR  = nan(length(ampThresholdList), length(INprobList));   % closed-form + K & E corrections

fprintf('Running  (%d T points, %d MC samples each) ...\n', ...
        length(ampThresholdList), Msample);

for idxINp = 1:length(INprobList)
    INp = INprobList(idxINp);

    pList = [1 - INp; INp];

    t0 = tic;

    parfor TIdx = 1:length(ampThresholdList)
        T = ampThresholdList(TIdx);
        expList = exp(-T .^ 2 ./ powerList);

        KBlank = 1 - sum(pList .* (1 + T .^ 2 ./ powerList) .* expList, 1);
        EoutBlank = sum(pList .* (powerList - (T .^ 2 + powerList) .* expList), 1);
        gammaBlank(TIdx, idxINp) = (EoutBlank / (powerS * abs(KBlank) ^ 2) - 1) ^ -1;

        KPGIRSig = sum(pList .* powerS .* (1 + powerS .* T .^ 2 ./ powerList .^ 2) .* expList, 1);
        EoutPGIRSig = KPGIRSig;

        U = sum(pList .* expList, 1);
        [powerE, errorSigCon] = reconErr(fftSize, nullIdx, T, INp, ...
                                 powerS, powerW, powerG, ...
                                 iterCount, Msample, OSFactor);
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
for idxINp = 1:length(INprobList)
    plot(ampThresholdList, snr_db(gammaPGIR(:, idxINp)), '-',  Color=colors(idxINp,:), ...
        DisplayName=sprintf('p = %g', INprobList(idxINp)), LineWidth=linewidth);
    plot(ampThresholdList, snr_db(gammaBlank(:, idxINp)), '--', Color=colors(idxINp,:), ...
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

function [powerE, errorSigCon] = reconErr(fftSize, nullMask, T, p, powerS, powerW, powerG, iterCount, Msample, OSFactor)

    transMask = zeros(fftSize, 1);
    transMask(nullMask) = 1;

    totabssquE = 0;
    toterrorSig = 0;
    cnt     = 0;

    for m = 1:Msample
        % ---- OFDM signal (with null subcarriers) ----------------------
        X = (randn(fftSize,1) + 1j*randn(fftSize,1)) / sqrt(2);
        X(nullMask) = 0;
        x = sqrt(OSFactor) .* sqrt(fftSize) .* ifft(X);

        % ---- Bernoulli-Gaussian noise --------------------------------
        w = (randn(fftSize,1) + 1j*randn(fftSize,1)) / sqrt(2) * sqrt(powerW);
        g = (randn(fftSize,1) + 1j*randn(fftSize,1)) / sqrt(2) * sqrt(powerG);
        b = rand(fftSize,1) < p;
        u = w + b.*g;
        r = x + u;

        unkDataMask = abs(r) > T;
        dataMask = ~unkDataMask;
        if ~any(unkDataMask),  continue;  end

        % ---- PGIR  xhat^(i+1) = P_D(P_T(xhat^(i))),  xhat^(0) = 0 ----
        xhat = PGIR(r, x, dataMask, transMask, iterCount);

        error = xhat(unkDataMask) - x(unkDataMask);
        totabssquE = totabssquE + sum(abs(error).^2);
        toterrorSig = toterrorSig + sum(error .* conj(x(unkDataMask)));
        cnt     = cnt + sum(unkDataMask);
    end

    if cnt == 0
        powerE = 0;  errorSigCon = 0;
    else
        powerE      = totabssquE / cnt;
        errorSigCon = toterrorSig / cnt;
    end

end