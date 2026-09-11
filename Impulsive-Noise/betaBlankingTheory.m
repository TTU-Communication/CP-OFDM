function out = betaBlankingTheory(Tvec, p, varargin)
%BETABLANKINGTHEORY  Theoretical null-tone distortion fraction beta(T) for blanking.
%
%   out = betaBlankingTheory(Tvec, p)
%   out = betaBlankingTheory(Tvec, p, 'Name', Value, ...)
%
%   Computes, without Monte Carlo, the fraction of distortion power falling on
%   the null (zero-padded) subcarriers at the output of the blanking
%   nonlinearity y_n = r_n * 1{|r_n| <= T}, under the oversampled OFDM +
%   Bernoulli-Gaussian impulsive-noise model of the manuscript:
%
%       beta(T) = sum_{k in nullSet} E|V_k|^2 / sum_k E|V_k|^2,   V = F*v,
%       v_n = y_n - K*x_n,   K = E{y_n x_n^*}/sigma_x^2,
%
%   so that  gamma_act(T) = gamma(T) / (1 - beta(T)).
%
%   Conventions (match the manuscript, NOT Zhidkov's half-moment convention):
%       sigma_x^2 = E|x_n|^2 = 1,  sigma_w^2 = 10^(-SNRdB/10),
%       sigma_g^2 = 10^(-SINRdB/10),  F unitary (F'*F = I).
%
%   Theory implemented (exact under: Gaussian signal, i.i.d. BG noise,
%   memoryless blanking; series truncated at order Q with c_q >= 0):
%       s_e^2 = sigma_x^2 + sigma_e^2 ,  u_e = T^2/s_e^2 ,  e in {noIN, IN}
%       M_0(u) = 1-(1+u)exp(-u)
%       M_q(u) = (u^2/q) * L_{q-1}^{(2)}(u) * exp(-u),  q >= 1
%       K       = sum_e P_e M_0(u_e)                       (Zhidkov (11), full-moment)
%       E|y|^2  = sum_e P_e s_e^2 M_0(u_e)                 (Zhidkov (13), full-moment)
%       sigma_v^2 = E|y|^2 - K^2
%       c_q     = [ sum_e P_e M_q(u_e) / (sqrt(q+1) s_e^(2q)) ]^2
%       R_v(tau)= sum_{q>=1} c_q rho(tau)|rho(tau)|^(2q),  tau ~= 0
%       R_v(0)  = sigma_v^2   (white remainder sigma_v^2 - sum_q c_q at lag 0)
%       Phi_v   = FFT_tau{R_v},   beta = sum_null Phi_v / sum_all Phi_v
%   with rho(tau) = (1/Nact) sum_{k in activeSet} exp(j 2 pi k tau / N).
%
%   INPUTS
%       Tvec : vector of blanking thresholds (amplitude units, sigma_x = 1)
%       p    : impulse probability
%
%   NAME-VALUE OPTIONS (defaults in brackets)
%       'N'          [256]   FFT size
%       'Los'        [4]     oversampling factor (Nact = N/Los)
%       'SNRdB'      [25]    10*log10(1/sigma_w^2)
%       'SINRdB'     [-15]   10*log10(1/sigma_g^2)
%       'Q'          [60]    series truncation order (Q=40 already ~1e-6)
%       'ActiveMask' []      logical N x 1 over FFT bins k = 0..N-1
%                            (true = data tone). Default: DC-centred band,
%                            k in [0..Nact/2-1] U [N-Nact/2..N-1].
%                            *** Replace with the complement of pgirNullSet
%                            to match the simulator exactly. ***
%       'SelfTest'   [false] run a built-in Monte Carlo at each T and print
%                            a comparison table (CPU, chunked)
%       'NSym'       [20000] symbols per Monte Carlo point
%       'Seed'       [1]     RNG seed for the self-test
%
%   OUTPUT (struct)
%       .T, .beta      : theoretical beta(T)      (series + exact discrete FFT)
%       .betaLO        : leading-order closed form 3/4 - (5/12) c_1 / sigma_v^2
%       .K, .sigmaV2   : attenuation factor and distortion power (closed form)
%       .gammaPooled   : K^2 / sigma_v^2            (time-domain gamma, eq.(18)-type)
%       .gammaAct      : gammaPooled ./ (1 - beta)  (data-subcarrier SNR)
%       .cq            : Q x nT matrix of series coefficients
%       .white         : white remainder sigma_v^2 - sum_q c_q
%       .diag          : minPhi (PSD positivity), parseval (should be ~1),
%                        nullFracQ1 (discrete out-of-band fraction of the
%                        q=1 shape; continuum value is 1/3)
%       .betaMC, .KMC, .sigmaV2MC : filled when 'SelfTest' is true
%
%   EXAMPLE
%       T   = (0.2:0.05:4.5).';
%       out = betaBlankingTheory(T, 0.01);
%       figure; plot(T, out.beta, 'LineWidth', 1.5); hold on;
%       plot(T, out.betaLO, ':');  yline(0.75, '--');
%       xlabel('T'); ylabel('\beta(T)'); legend('series', 'q=1 closed form', '3/4');
%       % overlay the measured beta from the calibration pass here
%
%   NOTES FOR COMPARISON WITH THE SIMULATOR
%     * The measured beta must use the SAME K convention: v = y - K*x with the
%       ensemble K (theoretical, or estimated by pooling all symbols). A
%       per-symbol regression K introduces an O(1/N) bias on the null tones.
%     * Noise must be ON in r (the retained noise on unblanked samples is part
%       of v). Modulation of the active tones should not matter (Gaussian, or
%       any QAM order).
%     * beta <= 3/4 everywhere and beta -> 3/4 as T -> inf are structural
%       predictions of the theory (c_q >= 0). A measured beta significantly
%       above 3/4 falsifies the derivation.

% ------------------------- options -------------------------
ip = inputParser;
ip.addParameter('N', 256);
ip.addParameter('Los', 4);
ip.addParameter('SNRdB', 25);
ip.addParameter('SINRdB', -15);
ip.addParameter('Q', 60);
ip.addParameter('ActiveMask', []);
ip.addParameter('SelfTest', false);
ip.addParameter('NSym', 20000);
ip.addParameter('Seed', 1);
ip.parse(varargin{:});
o = ip.Results;

N    = o.N;
Nact = N / o.Los;
assert(mod(Nact, 1) == 0, 'N must be divisible by Los.');
Q    = o.Q;

sw2 = 10^(-o.SNRdB / 10);      % sigma_w^2  (full second moment)
sg2 = 10^(-o.SINRdB / 10);     % sigma_g^2

if isempty(o.ActiveMask)
    kk = (0:N-1).';
    activeMask = (kk < Nact/2) | (kk >= N - Nact/2);   % DC-centred band
else
    activeMask = logical(o.ActiveMask(:));
    assert(numel(activeMask) == N && nnz(activeMask) == Nact, ...
        'ActiveMask must be N x 1 logical with N/Los true entries.');
end
nullMask = ~activeMask;

% signal circular autocorrelation rho(tau), tau = 0..N-1 (complex in general)
rho    = ifft(double(activeMask)) * N / Nact;          % rho(1) = 1  (tau = 0)
absr2  = abs(rho).^2;

% event bookkeeping: e = 1 (no impulse), e = 2 (impulse)
s2 = [1 + sw2, 1 + sw2 + sg2];                         % Rayleigh parameters s_e^2
Pe = [1 - p, p];

Tvec = Tvec(:);
nT   = numel(Tvec);

out = struct();
out.T           = Tvec;
out.beta        = zeros(nT, 1);
out.betaLO      = zeros(nT, 1);
out.K           = zeros(nT, 1);
out.sigmaV2     = zeros(nT, 1);
out.cq          = zeros(Q, nT);
out.white       = zeros(nT, 1);
out.diag.minPhi   = zeros(nT, 1);
out.diag.parseval = zeros(nT, 1);

% diagnostic: discrete out-of-band fraction of the q = 1 shape (continuum: 1/3)
psi1 = real(fft(rho .* absr2));
out.diag.nullFracQ1 = sum(psi1(nullMask)) / sum(psi1);

for it = 1:nT
    T  = Tvec(it);
    u  = T^2 ./ s2;                                    % 1 x 2
    ex = exp(-u);

    M0  = 1 - (1 + u) .* ex;
    K   = Pe * M0.';
    Ey2 = Pe * (s2 .* M0).';
    sv2 = Ey2 - K^2;

    % ---- series: c_q and shaped correlation ----
    Rv   = zeros(N, 1);
    wq   = rho .* absr2;                               % rho |rho|^2   (q = 1)
    Lm1  = zeros(1, 2);                                % L_{-1}^{(2)}
    Lc   = ones(1, 2);                                 % L_{0}^{(2)}
    csum = 0;
    for q = 1:Q
        Mq  = (u.^2 / q) .* Lc .* ex;                  % M_q(u_e), closed form
        Bq  = Pe * (Mq ./ (sqrt(q + 1) * s2.^q)).';
        cq  = Bq^2;
        out.cq(q, it) = cq;
        csum = csum + cq;
        Rv   = Rv + cq * wq;
        % advance to next order
        wq   = wq .* absr2;                            % rho |rho|^(2(q+1))
        Lnew = ((2*q + 1 - u) .* Lc - (q + 1) .* Lm1) / q;   % L_q^{(2)}
        Lm1  = Lc;
        Lc   = Lnew;
    end

    white = sv2 - csum;                                % >= 0 by theory
    Rv(1) = Rv(1) + white;                             % R_v(0) = sigma_v^2

    Phi  = real(fft(Rv));                              % E|V_k|^2, k = 0..N-1
    beta = sum(Phi(nullMask)) / sum(Phi);

    out.beta(it)    = beta;
    out.betaLO(it)  = 3/4 - (5/12) * out.cq(1, it) / sv2;
    out.K(it)       = K;
    out.sigmaV2(it) = sv2;
    out.white(it)   = white;
    out.diag.minPhi(it)   = min(Phi);
    out.diag.parseval(it) = sum(Phi) / (N * sv2);
end

out.gammaPooled    = out.K.^2 ./ out.sigmaV2;
out.gammaAct       = out.gammaPooled ./ (1 - out.beta);
out.gammaPooled_dB = 10 * log10(out.gammaPooled);
out.gammaAct_dB    = 10 * log10(out.gammaAct);

% ------------------------- optional Monte Carlo self-test -------------------------
if o.SelfTest
    rng(o.Seed);
    out.betaMC    = zeros(nT, 1);
    out.KMC       = zeros(nT, 1);
    out.sigmaV2MC = zeros(nT, 1);
    fprintf('%6s  %10s  %10s  %9s  %8s  %8s\n', ...
        'T', 'beta_th', 'beta_mc', 'rel.err', 'K_th', 'K_mc');
    for it = 1:nT
        [bmc, Kmc, svmc] = localMC(Tvec(it), p, N, Nact, activeMask, ...
                                   sw2, sg2, o.NSym);
        out.betaMC(it)    = bmc;
        out.KMC(it)       = Kmc;
        out.sigmaV2MC(it) = svmc;
        fprintf('%6.3f  %10.5f  %10.5f  %+8.1e  %8.5f  %8.5f\n', ...
            Tvec(it), out.beta(it), bmc, out.beta(it)/bmc - 1, out.K(it), Kmc);
    end
end
end

% =====================================================================
function [beta, Kmc, sv2] = localMC(T, p, N, Nact, activeMask, sw2, sg2, nsym)
%LOCALMC  Direct Monte Carlo of the model (Gaussian active-tone symbols).
chunk  = 4000;
PhiAcc = zeros(1, N);
yxAcc  = 0;  xxAcc = 0;
Los    = N / Nact;
nleft  = nsym;
% Two passes with replayed RNG: pass 1 forms the pooled K, pass 2 uses that
% same K in v = y - K x (same convention as the theory).
% --- pass 1: pooled K ---
rngState = rng;                    % replay identical noise in pass 2
while nleft > 0
    nb = min(chunk, nleft);  nleft = nleft - nb;
    [x, y] = genChunk(nb, T, p, N, Nact, Los, activeMask, sw2, sg2);
    yx    = y .* conj(x);
    yxAcc = yxAcc + sum(yx(:));
    xxAcc = xxAcc + sum(abs(x(:)).^2);
end
Kmc = real(yxAcc / xxAcc);
% --- pass 2: distortion spectrum with pooled K ---
rng(rngState);
nleft = nsym;  sv2Acc = 0;  ntot = 0;
while nleft > 0
    nb = min(chunk, nleft);  nleft = nleft - nb;
    [x, y] = genChunk(nb, T, p, N, Nact, Los, activeMask, sw2, sg2);
    v  = y - Kmc * x;
    V  = fft(v, [], 2) / sqrt(N);
    PhiAcc = PhiAcc + sum(abs(V).^2, 1);
    sv2Acc = sv2Acc + sum(abs(v(:)).^2);
    ntot   = ntot + nb;
end
Phi  = PhiAcc / ntot;
beta = sum(Phi(~activeMask.')) / sum(Phi);
sv2  = sv2Acc / (ntot * N);
end

function [x, y] = genChunk(nb, T, p, N, Nact, Los, activeMask, sw2, sg2)
X = zeros(nb, N);
X(:, activeMask.') = (randn(nb, Nact) + 1j * randn(nb, Nact)) * sqrt(Los / 2);
x = ifft(X, [], 2) * sqrt(N);                          % unitary IDFT, E|x_n|^2 = 1
w = (randn(nb, N) + 1j * randn(nb, N)) * sqrt(sw2 / 2);
b = rand(nb, N) < p;
g = (randn(nb, N) + 1j * randn(nb, N)) * sqrt(sg2 / 2);
r = x + w + b .* g;
y = r .* (abs(r) <= T);
end
