function out = betaMemorylessTheory(Tvec, p, varargin)
%BETAMEMORYLESSTHEORY  Theoretical beta(T) for memoryless amplitude nonlinearities.
%
%   out = betaMemorylessTheory(Tvec, p, 'Scheme', 'deepclip', 'Mu', 2, ...)
%
%   Generalizes betaBlankingTheory to any phase-preserving amplitude
%   nonlinearity y_n = f(|r_n|) exp(j arg r_n) under the oversampled OFDM +
%   Bernoulli-Gaussian model (manuscript conventions, sigma_x^2 = 1):
%
%       beta(T) = sum_{k in nullSet} E|V_k|^2 / sum_k E|V_k|^2,  V = F(y - K x),
%       gamma_act(T) = gamma(T) / (1 - beta(T)).
%
%   Theory: per event e in {no-IN, IN} (Rayleigh parameter s_e^2, u = T^2/s_e^2)
%   the first-circular-harmonic Laguerre moments
%
%       G_q(e) = E{ a f(a) L_q^{(1)}(a^2/s_e^2) | e },   a = |r_n|,
%
%   give  K = sum_e P_e G_0(e)/s_e^2,   E|y|^2 = sum_e P_e E{f(a)^2 | e},
%   sigma_v^2 = E|y|^2 - K^2,  and the series coefficients
%
%       c_q = [ sum_e P_e G_q(e) / (sqrt(q+1) s_e^(2q+2)) ]^2 ,   q >= 1,
%
%   assembled exactly as in betaBlankingTheory:
%       R_v(tau) = sum_q c_q rho(tau)|rho(tau)|^(2q)  (tau ~= 0),
%       R_v(0)   = sigma_v^2,   Phi_v = FFT{R_v},
%       beta     = sum_null Phi_v / sum_all Phi_v.
%
%   G_q(e) and E{f^2} are evaluated by piecewise Gauss-Legendre quadrature in
%   t = a^2/s_e^2 (weight e^{-t}), with L_q^{(1)} by stable upward recurrence.
%   For blanking this reproduces the closed-form M_q path of
%   betaBlankingTheory to ~1e-8; for deep clipping the q = 0 moments match the
%   closed forms
%       K_e   = (1+mu) M_0(u) - mu M_0(u2) + (1+mu)(T/s_e) H(u, u2)
%       E|y|^2_e = s_e^2 M_0(u) + (1+mu)^2 T^2 (e^{-u} - e^{-u2})
%                  - 2 mu (1+mu) T s_e H(u, u2) + mu^2 s_e^2 (M_0(u2)-M_0(u))
%   with u2 = (1+1/mu)^2 u and
%       H(x1,x2) = sqrt(x1) e^{-x1} - sqrt(x2) e^{-x2}
%                  + (sqrt(pi)/2)(erf(sqrt(x2)) - erf(sqrt(x1))).
%
%   SCHEMES ('Scheme' option)
%     'blanking'  : f(a) = a (a<=T), 0 else
%     'clipping'  : f(a) = min(a, T)
%     'clipblank' : f(a) = a (a<=T), T (T<a<=Tb), 0 else, Tb = TbFactor*T
%     'deepclip'  : f(a) = a (a<=T), (1+Mu)T - Mu a (T<a<=(1+1/Mu)T), 0 else
%     'custom'    : user handle 'F' (vectorized f(a)) with breakpoints
%                   'KnotsA' (ascending amplitudes; f may be nonzero beyond
%                   the last knot, the tail is integrated to negligible mass)
%
%   NAME-VALUE OPTIONS (defaults)
%     'Scheme'   ['deepclip']    'Mu' [2]      'TbFactor' [1.4]
%     'F' []     'KnotsA' []     'Nodes' [400] (quadrature nodes per piece)
%     'N' [256]  'Los' [4]       'SNRdB' [25]  'SINRdB' [-15]   'Q' [60]
%     'ActiveMask' []  (logical N x 1 over FFT bins k = 0..N-1; default
%                       DC-centred band -- replace with the complement of
%                       pgirNullSet to match the simulator)
%     'SelfTest' [false]  'NSym' [20000]  'Seed' [1]
%
%   OUTPUT: struct with T, beta, betaLO, K, sigmaV2, gammaPooled, gammaAct
%   (+dB), cq, white, diag (minPhi / parseval / nullFracQ1), and betaMC /
%   KMC / sigmaV2MC when 'SelfTest' is on.
%
%   EXAMPLE
%     T = (0.2:0.05:4.5).';
%     dc = betaMemorylessTheory(T, 0.1, 'Scheme', 'deepclip', 'Mu', 2);
%     bl = betaMemorylessTheory(T, 0.1, 'Scheme', 'blanking');
%     plot(T, [dc.beta bl.beta]); yline(0.75, '--');
%     legend('deep clipping \mu = 2', 'blanking');
%
%   Notes: same comparison caveats as betaBlankingTheory (ensemble K in
%   v = y - K x, AWGN on, modulation-independent). beta <= 3/4 with
%   beta -> 3/4 as T -> inf remains structural (c_q >= 0).

% ------------------------- options -------------------------
ip = inputParser;
ip.addParameter('Scheme', 'deepclip');
ip.addParameter('Mu', 2);
ip.addParameter('TbFactor', 1.4);
ip.addParameter('F', []);
ip.addParameter('KnotsA', []);
ip.addParameter('Nodes', 400);
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

sw2 = 10^(-o.SNRdB / 10);
sg2 = 10^(-o.SINRdB / 10);

if isempty(o.ActiveMask)
    kk = (0:N-1).';
    activeMask = (kk < Nact/2) | (kk >= N - Nact/2);
else
    activeMask = logical(o.ActiveMask(:));
    assert(numel(activeMask) == N && nnz(activeMask) == Nact, ...
        'ActiveMask must be N x 1 logical with N/Los true entries.');
end
nullMask = ~activeMask;

rho   = ifft(double(activeMask)) * N / Nact;
absr2 = abs(rho).^2;

s2 = [1 + sw2, 1 + sw2 + sg2];
Pe = [1 - p, p];

[xg, wg] = gaussLegendre(o.Nodes);      % nodes/weights on [-1, 1]

Tvec = Tvec(:);
nT   = numel(Tvec);

out = struct();
out.scheme      = o.Scheme;
out.T           = Tvec;
out.beta        = zeros(nT, 1);
out.betaLO      = zeros(nT, 1);
out.K           = zeros(nT, 1);
out.sigmaV2     = zeros(nT, 1);
out.cq          = zeros(Q, nT);
out.white       = zeros(nT, 1);
out.diag.minPhi   = zeros(nT, 1);
out.diag.parseval = zeros(nT, 1);

psi1 = real(fft(rho .* absr2));
out.diag.nullFracQ1 = sum(psi1(nullMask)) / sum(psi1);

qv = (1:Q);

for it = 1:nT
    T = Tvec(it);
    K = 0; Ey2 = 0; atil = zeros(1, Q);
    for e = 1:2
        pieces = schemePieces(o, T, s2(e));
        [G, E2] = eventMoments(pieces, s2(e), Q, xg, wg);
        K    = K   + Pe(e) * G(1) / s2(e);
        Ey2  = Ey2 + Pe(e) * E2;
        atil = atil + Pe(e) * G(2:end) ./ (sqrt(qv + 1) .* s2(e).^(qv + 1));
    end
    sv2 = Ey2 - K^2;
    cq  = atil.^2;

    Rv = zeros(N, 1);
    wq = rho .* absr2;
    for q = 1:Q
        Rv = Rv + cq(q) * wq;
        wq = wq .* absr2;
    end
    white = sv2 - sum(cq);
    Rv(1) = Rv(1) + white;

    Phi = real(fft(Rv));
    out.beta(it)    = sum(Phi(nullMask)) / sum(Phi);
    out.betaLO(it)  = 3/4 - (5/12) * cq(1) / sv2;
    out.K(it)       = K;
    out.sigmaV2(it) = sv2;
    out.cq(:, it)   = cq(:);
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
    famp = ampHandle(o);
    out.betaMC    = zeros(nT, 1);
    out.KMC       = zeros(nT, 1);
    out.sigmaV2MC = zeros(nT, 1);
    fprintf('%6s  %10s  %10s  %9s  %8s  %8s\n', ...
        'T', 'beta_th', 'beta_mc', 'rel.err', 'K_th', 'K_mc');
    for it = 1:nT
        [bmc, Kmc, svmc] = localMC(famp, Tvec(it), p, N, Nact, activeMask, ...
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
function pieces = schemePieces(o, T, s2e)
%SCHEMEPIECES  Piecewise definition of f in t = a^2/s_e^2 (columns: t1, t2, f).
u    = T^2 / s2e;
tPad = 45;                                  % e^{-45} tail, negligible
switch lower(o.Scheme)
    case 'blanking'
        pieces = {0, u, @(a) a};
    case 'clipping'
        pieces = {0, u, @(a) a; u, u + tPad, @(a) T + 0*a};
    case 'clipblank'
        ub = (o.TbFactor * T)^2 / s2e;
        pieces = {0, u, @(a) a; u, ub, @(a) T + 0*a};
    case 'deepclip'
        mu = o.Mu;
        u2 = (1 + 1/mu)^2 * u;
        pieces = {0, u, @(a) a; u, u2, @(a) (1 + mu)*T - mu*a};
    case 'custom'
        assert(~isempty(o.F), 'custom scheme requires the ''F'' handle.');
        ka = sort(o.KnotsA(:)).';
        tk = [0, (ka.^2) / s2e];
        tk = [tk, tk(end) + tPad];
        pieces = cell(numel(tk) - 1, 3);
        for i = 1:numel(tk) - 1
            pieces(i, :) = {tk(i), tk(i + 1), o.F};
        end
    otherwise
        error('Unknown scheme %s', o.Scheme);
end
end

function famp = ampHandle(o)
%AMPHANDLE  f(a) as a single vectorized handle (for the Monte Carlo).
switch lower(o.Scheme)
    case 'blanking'
        famp = @(a, T) a .* (a <= T);
    case 'clipping'
        famp = @(a, T) min(a, T);
    case 'clipblank'
        tb = o.TbFactor;
        famp = @(a, T) a .* (a <= T) + T .* (a > T & a <= tb*T);
    case 'deepclip'
        mu = o.Mu;
        famp = @(a, T) a .* (a <= T) + ...
            ((1 + mu)*T - mu*a) .* (a > T & a <= (1 + 1/mu)*T);
    case 'custom'
        famp = @(a, T) o.F(a);
end
end

function [G, Ey2] = eventMoments(pieces, s2e, Q, xg, wg)
%EVENTMOMENTS  G_q = E{a f(a) L_q^{(1)}(t)}, Ey2 = E{f(a)^2}, t = a^2/s_e^2.
s   = sqrt(s2e);
G   = zeros(1, Q + 1);
Ey2 = 0;
for i = 1:size(pieces, 1)
    t1 = pieces{i, 1};  t2 = pieces{i, 2};  f = pieces{i, 3};
    if t2 <= t1, continue; end
    t  = 0.5*(t2 - t1)*xg + 0.5*(t2 + t1);
    w  = 0.5*(t2 - t1)*wg;
    a  = s * sqrt(t);
    fa = f(a);
    ex = exp(-t);
    Ey2  = Ey2 + sum(w .* fa.^2 .* ex);
    base = w .* a .* fa .* ex;
    Lm1 = zeros(size(t));  Lc = ones(size(t));      % L_0^{(1)}
    for q = 0:Q
        G(q + 1) = G(q + 1) + sum(base .* Lc);
        Ln  = ((2*q + 2 - t) .* Lc - (q + 1) .* Lm1) / (q + 1);
        Lm1 = Lc;  Lc = Ln;
    end
end
end

function [x, w] = gaussLegendre(n)
%GAUSSLEGENDRE  Golub-Welsch nodes/weights on [-1, 1].
k = (1:n-1);
b = k ./ sqrt(4*k.^2 - 1);
J = diag(b, 1) + diag(b, -1);
[V, D] = eig(J);
[x, idx] = sort(diag(D));
w = (2 * V(1, idx).^2).';
x = x(:);  w = w(:);
end

function [beta, Kmc, sv2] = localMC(famp, T, p, N, Nact, activeMask, sw2, sg2, nsym)
%LOCALMC  Direct Monte Carlo (Gaussian active-tone symbols), pooled-K convention.
chunk  = 4000;
PhiAcc = zeros(1, N);
yxAcc  = 0;  xxAcc = 0;
Los    = N / Nact;
nleft  = nsym;
rngState = rng;
while nleft > 0
    nb = min(chunk, nleft);  nleft = nleft - nb;
    [x, y] = genChunk(famp, nb, T, p, N, Nact, Los, activeMask, sw2, sg2);
    yx    = y .* conj(x);
    yxAcc = yxAcc + sum(yx(:));
    xxAcc = xxAcc + sum(abs(x(:)).^2);
end
Kmc = real(yxAcc / xxAcc);
rng(rngState);
nleft = nsym;  sv2Acc = 0;  ntot = 0;
while nleft > 0
    nb = min(chunk, nleft);  nleft = nleft - nb;
    [x, y] = genChunk(famp, nb, T, p, N, Nact, Los, activeMask, sw2, sg2);
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

function [x, y] = genChunk(famp, nb, T, p, N, Nact, Los, activeMask, sw2, sg2)
X = zeros(nb, N);
X(:, activeMask.') = (randn(nb, Nact) + 1j * randn(nb, Nact)) * sqrt(Los / 2);
x = ifft(X, [], 2) * sqrt(N);
w = (randn(nb, N) + 1j * randn(nb, N)) * sqrt(sw2 / 2);
b = rand(nb, N) < p;
g = (randn(nb, N) + 1j * randn(nb, N)) * sqrt(sg2 / 2);
r = x + w + b .* g;
y = famp(abs(r), T) .* exp(1j * angle(r));
end
