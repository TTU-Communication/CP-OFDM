% BLANKING_RAYLEIGH_NOMINALMMSE_CLOSEDFORM
% Deterministic analytical Output-SNR curves for
%   OFDM -> L-tap Rayleigh multipath -> blanking -> FFT -> nominal MMSE EQ.
%
% IMPORTANT:
%   The MMSE equalizer here DOES NOT include impulsive-noise power and does
%   NOT include post-blanking residual power.  It is designed only from the
%   physical channel H_k and background AWGN variance sigma_w2:
%
%       W_k^nom = H_k^* P_X / ( |H_k|^2 P_X + sigma_w2 ).
%
% No Monte-Carlo symbols, noise, channel draws, or random-number generation
% are used.  All Rayleigh averages are deterministic Gauss-Laguerre /
% Gauss-Jacobi quadratures.
%
% System parameters:
%   N = 256, Nnull = 32 (16 left + 16 right guard-band nulls),
%   L = 8, SNR = 25 dB, SINR = -15 dB,
%   p = {0.001, 0.01, 0.1}, T in [0.1, 15].
%
% Power convention:
%   sigma_x2 = E{|x_n|^2} is the average time-domain OFDM power.
%   P_X = E{|X_k|^2} on an active subcarrier.
%
% Closed-form/analytical approximation retained from the previous model:
%   beta_h^2 = sigma_x2 * eta, eta = ||h||^2 ~ Gamma(L,1/L),
%   |H_k|^2 = eta*G, with G/L ~ Beta(1,L-1),
%   S_D,k ~= sigma_d^2 (white-residual approximation).
%
% Therefore, this file changes ONLY the equalizer model relative to the
% previous post-blanking-LMMSE version.  It does not claim to remove the
% remaining beta_h^2 and white-residual approximations.
clc; clear;

%% 1) System parameters
N       = 256;
Nact    = 224;
L_ch    = 8;
SNRdB   = 25;
SINRdB  = -15;
pList   = [1e-3, 1e-2, 1e-1];

% 16 null tones at each edge of the FFT grid.
Nnull = N - Nact;
nullIdx   = [1:(Nnull/2), (N-Nnull/2+1):N]; %#ok<NASGU>
activeIdx = setdiff(1:N, nullIdx); %#ok<NASGU>

% Full second-moment convention.
P_X      = 1;
sigma_x2 = Nact / N * P_X;       % 224/256 = 0.875
sigma_w2 = sigma_x2 / 10^(SNRdB/10);
sigma_g2 = sigma_x2 / 10^(SINRdB/10);

T = linspace(0.1, 15, 401);

%% 2) Deterministic Rayleigh-fading quadrature
% eta = z/L, z ~ Gamma(shape=L, scale=1).
nEta = 192;
[z, wLag] = gaussLaguerre(nEta, L_ch-1);
eta       = z / L_ch;
wEta      = wLag / gamma(L_ch);               % E_eta{.}

% % G = L*v, v ~ Beta(1,L-1).
% nG = 48;
% [xJac, wJac] = gaussJacobi(nG, L-2, 0);
% v  = (xJac + 1)/2;
% G  = L*v;
% wG = wJac * (L-1)/2^(L-1);                 % E_G{.}

% Conditional distribution of G given eta:
% |H_k|^2 = eta * G.

if L_ch == 1
    % Flat Rayleigh fading:
    % H_k = h_0, eta = |h_0|^2, hence G = 1 exactly.
    nG = 1;
    G  = 1;
    wG = 1;

else
    % L >= 2:
    % G/L ~ Beta(1, L-1).
    nG = 192;

    [xJac, wJac] = gaussJacobi(nG, L_ch-2, 0);

    v  = (xJac + 1) / 2;
    G  = L_ch * v;

    % Normalize the Jacobi rule into the Beta(1,L-1) expectation.
    wG = wJac * (L_ch - 1) / 2^(L_ch - 1);
end

[ETA, GG] = ndgrid(eta, G);
MU        = ETA .* GG;                     % MU = |H_k|^2
WQ        = wEta * wG.';                   % product quadrature weights

snrOut_dB = zeros(numel(pList), numel(T));

%% 3) Closed-form blanking moments + nominal-MMSE equalization
for ip = 1:numel(pList)
    p = pList(ip);

    for it = 1:numel(T)
        thr = T(it);

        % 3.1 Conditional blanking input powers.
        % beta_h2 is retained as the scalar eta-conditioned approximation.
        beta_h2 = sigma_x2 * eta;
        R0      = beta_h2 + sigma_w2;
        R1      = beta_h2 + sigma_w2 + sigma_g2;

        e0 = exp(-(thr^2)./R0);
        e1 = exp(-(thr^2)./R1);

        % 3.2 Exact scalar blanking Bussgang gain K0(beta_h2).
        K0 = 1 ...
           - (1-p) .* (1 + thr^2./R0) .* e0 ...
           - p       .* (1 + thr^2./R1) .* e1;

        % 3.3 Exact scalar blanker output power E{|y_n|^2 | eta}.
        Eblank = (1-p) .* (R0 - (thr^2 + R0).*e0) ...
               + p     .* (R1 - (thr^2 + R1).*e1);

        % 3.4 Scalar Bussgang residual power.
        sigma_d2 = real(Eblank - K0.^2 .* beta_h2);
        sigma_d2 = max(sigma_d2, 1e-15);   % numerical protection only

        % Expand eta-only terms over G nodes.
        K0m = K0 * ones(1, nG);
        Dm  = sigma_d2 * ones(1, nG);

        % 3.5 NOMINAL MMSE EQ: IN and blanking residual are NOT included.
        % W_k^nom = H_k^* P_X/(MU*P_X + sigma_w2).
        denNom = MU * P_X + sigma_w2;

        % Useful gain from X_k to Xhat_k:
        % a_k = W_k^nom * K0*H_k = K0*MU*P_X/(MU*P_X + sigma_w2).
        a = K0m .* MU .* P_X ./ denNom;

        % Equalized blanking-residual power under white-residual approx:
        % |W_k^nom|^2 sigma_d2.
        nEq = MU .* (P_X^2) .* Dm ./ (denNom.^2);

        % % Total equalizer-output power.
        % Pout = P_X .* abs(a).^2 + nEq;
        % 
        % % 3.6 Pooled Bussgang Output SNR, deterministic channel average.
        % Abar = sum(sum(WQ .* a));
        % Qbar = sum(sum(WQ .* Pout));
        % 
        % % The denominator includes fading-induced gain variation and
        % % residual distortion after the nominal MMSE equalizer.
        % denom = max(real(Qbar - P_X*abs(Abar)^2), realmin);
        % gammaOut = P_X*abs(Abar)^2 / denom;
        % Full-band average output power.
        % Useful signal exists only on Nact active tones.
        % Equalized noise/distortion exists on all N tones.
        PoutFull = sigma_x2 .* abs(a).^2 + nEq;

        Abar = sum(sum(WQ .* a));
        Qbar = sum(sum(WQ .* PoutFull));

        denom = max(real(Qbar - sigma_x2 * abs(Abar)^2), realmin);
        gammaOut = sigma_x2 * abs(Abar)^2 / denom;

        snrOut_dB(ip,it) = 10*log10(gammaOut);
    end
end

%% 4) Plot
figure;
hold on; grid on; box on;
plot(T, snrOut_dB(1,:), 'LineWidth', 1.8);
plot(T, snrOut_dB(2,:), 'LineWidth', 1.8);
plot(T, snrOut_dB(3,:), 'LineWidth', 1.8);

xlim([0 15]);
ylim([-25 25]);
xlabel('Blanking threshold, T');
ylabel('Output SNR (dB)');
title(sprintf(['Closed-form blanking + nominal MMSE (AWGN only), ', ...
    'N=%d, N_{null}=%d, L=%d, SNR=%g dB, SINR=%g dB'], ...
    N, Nnull, L_ch, SNRdB, SINRdB));
legend('p = 0.001', 'p = 0.01', 'p = 0.1', 'Location', 'southwest');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

%% 5) Deterministic optimum summary
fprintf('\nDeterministic analytical results: blanking + nominal MMSE (AWGN only)\n');
for ip = 1:numel(pList)
    [peak, idx] = max(snrOut_dB(ip,:));
    fprintf('p = %-6g : T_opt = %8.5f, max Output SNR = %8.4f dB\n', ...
        pList(ip), T(idx), peak);
end

%% function handler
function [x, w] = gaussLaguerre(n, alpha)
% Generalized Gauss-Laguerre quadrature:
% integral_0^inf x^alpha exp(-x) f(x) dx ~= sum_i w_i f(x_i).

    k = (0:n-1).';
    d = 2*k + alpha + 1;
    e = sqrt((1:n-1).' .* ((1:n-1).' + alpha));
    J = diag(d) + diag(e,1) + diag(e,-1);

    [V, D] = eig((J + J.')/2);
    [x, idx] = sort(diag(D));
    V = V(:, idx);
    w = gamma(alpha+1) * (V(1,:).^2).';
end

% function [x, w] = gaussJacobi(n, alpha, betaPar)
% % Gauss-Jacobi quadrature on [-1,1] with weight
% % (1-x)^alpha (1+x)^betaPar.
% 
% ab = alpha + betaPar;
% k  = (0:n-1).';
% 
% d = (betaPar^2 - alpha^2) ./ ((2*k + ab) .* (2*k + ab + 2));
% 
% m = (0:n-2).';
% e = 2 ./ (2*m + ab + 2) .* sqrt( ...
%     ((m+1) .* (m+1+ab) .* (m+1+alpha) .* (m+1+betaPar)) ./ ...
%     ((2*m+ab+1) .* (2*m+ab+3)) );
% 
% J = diag(d) + diag(e,1) + diag(e,-1);
% [V, D] = eig((J + J.')/2);
% [x, idx] = sort(diag(D));
% V = V(:, idx);
% 
% mu0 = 2^(ab+1) * beta(alpha+1, betaPar+1);
% w = mu0 * (V(1,:).^2).';
% end

function [x, w] = gaussJacobi(n, alpha, betaPar)
% Gauss-Jacobi quadrature on [-1,1] with weight
% (1-x)^alpha (1+x)^betaPar.
%
% Requires alpha > -1 and betaPar > -1.

    if alpha <= -1 || betaPar <= -1
        error('gaussJacobi:InvalidParameters', ...
            'Require alpha > -1 and betaPar > -1.');
    end

    ab = alpha + betaPar;
    k  = (0:n-1).';

    % Diagonal recurrence coefficients.
    d = zeros(n, 1);

    if abs(ab) < 1e-14
        % Removable 0/0 singularity at k = 0:
        % lim_{a+b -> 0} (b^2-a^2)/[(a+b)(a+b+2)]
        % = (b-a)/(a+b+2).
        d(1) = (betaPar - alpha) / (ab + 2);

        if n > 1
            kk = k(2:end);
            d(2:end) = (betaPar^2 - alpha^2) ./ ...
                ((2*kk + ab) .* (2*kk + ab + 2));
        end

    else
        d = (betaPar^2 - alpha^2) ./ ...
            ((2*k + ab) .* (2*k + ab + 2));
    end

    % Off-diagonal recurrence coefficients.
    m = (0:n-2).';

    e = 2 ./ (2*m + ab + 2) .* sqrt( ...
        ((m+1) .* (m+1+ab) .* (m+1+alpha) .* (m+1+betaPar)) ./ ...
        ((2*m+ab+1) .* (2*m+ab+3)) );

    J = diag(d) + diag(e,1) + diag(e,-1);

    [V, D] = eig((J + J.') / 2);
    [x, idx] = sort(diag(D));
    V = V(:, idx);

    mu0 = 2^(ab + 1) * beta(alpha + 1, betaPar + 1);
    w   = mu0 * (V(1,:).^2).';
end
