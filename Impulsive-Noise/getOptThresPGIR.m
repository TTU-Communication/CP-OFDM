function [Topt, gammaOpt_dB, info] = getOptThresPGIR(Px, Pw, Pg, p, N, nullIdx, varargin)
%OPTIMALTHRESHOLDPGIR_MC_MANUSCRIPT  Optimize a PGIR threshold using the
%manuscript K_PGIR and E_PGIR decomposition.
%
%   [Topt,gammaOpt_dB,info] = optimalThresholdPGIR_mc_manuscript( ... )
%   evaluates, for every threshold T,
%
%       K_PGIR(T) = K_Blank(T) + (R_x(T) + DeltaK_PGIR(T))/Px,
%
%       E_PGIR(T) = E_Blank(T) + R_x(T) + DeltaE_PGIR(T),
%
%   where
%
%       DeltaK_PGIR = P(C) E{ e_n x_n^* | C },
%       DeltaE_PGIR = P(C) E{ |e_n|^2 | C }
%                      + 2 Re{DeltaK_PGIR}.
%
%   The two conditional error moments are estimated by the preserved
%   subfunction reconErr.  This is the same decomposition as manuscript
%   equations (31), (34), and (35), under the full-second-moment convention
%
%       Px = E{|x_n|^2}, Pw = E{|w_n|^2}, Pg = E{|g_n|^2}.
%
%   The output SNR is calculated from manuscript K_PGIR and E_PGIR as
%
%       gamma(T) = Px |K_PGIR(T)|^2 / (E_PGIR(T) - Px |K_PGIR(T)|^2).
%
%   IMPORTANT
%   ---------
%   1. The closed-form blanking and detected-signal terms require the
%      null-tone / zero-mean circular-Gaussian OFDM model in the manuscript.
%      Hence this implementation accepts null tones but rejects nonzero known
%      pilots in the manuscript-formula path.
%   2. nIter = Inf (default) evaluates the PGIR fixed point when the reduced
%      system is numerically well conditioned.  A problematic realization is
%      NOT discarded; it uses a finite-iteration fallback and is reported.
%   3. reconErr reconstructs xhat from r, the detection mask, and the null
%      constraint only.  It never initializes the reconstruction error using
%      the unknown transmitted x.
%
%   Required inputs
%   ---------------
%   Px,Pw,Pg : full second-moment powers.
%   p        : Bernoulli impulse-occurrence probability.
%   N        : FFT size.
%   nullIdx  : 1-based indices of transform-domain null subcarriers.
%
%   Name-value options
%   ------------------
%   'Tmin','Tmax','targetStep' : threshold grid.  Defaults are scaled by
%                                 sqrt(Px): [0.1, 8, 0.1]*sqrt(Px).
%   'M'                : Monte-Carlo OFDM blocks per seed.        (1000)
%   'OSFactor'         : oversampling normalization factor.           (1)
%                        Used only when generating normalized OFDM
%                        Monte-Carlo signals; PGIR projections and
%                        analytic K/E expressions are unchanged.
%   'seed'             : first random seed.                          (0)
%   'nSeeds'           : independent CRN pools combined per T.        (1)
%   'nIter'            : nonnegative integer, or Inf.              (Inf)
%   'initialization'   : PGIR xhat^(0): 'zero' or 'data'.         ('zero')
%   'steadyStateFallbackIter' : iteration count for ill-conditioned
%                                nIter=Inf realizations.              (50)
%   'rcondTol'         : fixed-point solve threshold.              (1e-10)
%   'pilotIdx','pilotVal' : retained for interface compatibility only.
%                           Nonempty pilots are rejected because the
%                           manuscript closed forms do not cover them.
%   'verifyDirectMC'   : also compute direct simulated K/E/y SNR as a
%                        diagnostic check of the decomposition.      (true)
%
%   The returned info structure includes K_Blank, E_Blank, R_x,
%   DeltaK_PGIR, DeltaE_PGIR, K_PGIR, E_PGIR and the direct-MC diagnostic
%   values at Topt, plus their complete threshold-grid traces.
%
%   See also theorem_suboptimal_PGIR.m, reconErr.

    ip = inputParser;
    ip.addParameter('Tmin', []);
    ip.addParameter('Tmax', []);
    ip.addParameter('targetStep', []);
    ip.addParameter('M', 1000);
    ip.addParameter('OSFactor', 1);
    ip.addParameter('seed', 0);
    ip.addParameter('nSeeds', 1);
    ip.addParameter('nIter', Inf);
    ip.addParameter('initialization', 'zero');
    ip.addParameter('steadyStateFallbackIter', 50);
    ip.addParameter('rcondTol', 1e-10);
    ip.addParameter('pilotIdx', []);
    ip.addParameter('pilotVal', []);
    ip.addParameter('verifyDirectMC', true);
    ip.parse(varargin{:});
    o = ip.Results;

    validateattributes(Px, {'numeric'}, {'scalar','real','finite','positive'});
    validateattributes(Pw, {'numeric'}, {'scalar','real','finite','nonnegative'});
    validateattributes(Pg, {'numeric'}, {'scalar','real','finite','nonnegative'});
    validateattributes(p,  {'numeric'}, {'scalar','real','finite','>=',0,'<=',1});
    validateattributes(N,  {'numeric'}, {'scalar','integer','>=',2});
    validateattributes(o.M, {'numeric'}, {'scalar','integer','>=',1});
    validateattributes(o.OSFactor, {'numeric'}, ...
        {'scalar','real','finite','positive'});
    validateattributes(o.nSeeds, {'numeric'}, {'scalar','integer','>=',1});
    validateattributes(o.steadyStateFallbackIter, {'numeric'}, ...
        {'scalar','integer','>=',0});
    validateattributes(o.rcondTol, {'numeric'}, {'scalar','real','finite','>=',0});

    if ~(isinf(o.nIter) || (isscalar(o.nIter) && isfinite(o.nIter) && ...
            o.nIter >= 0 && o.nIter == floor(o.nIter)))
        error('PGIRmanuscript:BadIterationCount', ...
            'nIter must be a nonnegative integer or Inf.');
    end

    o.initialization = lower(char(o.initialization));
    if ~ismember(o.initialization, {'zero','data'})
        error('PGIRmanuscript:BadInitialization', ...
            'initialization must be ''zero'' or ''data''.');
    end

    nullIdx = unique(nullIdx(:));
    if isempty(nullIdx)
        error('PGIRmanuscript:NoNullTone', ...
            'The manuscript formula requires at least one null subcarrier.');
    end
    if any(nullIdx < 1 | nullIdx > N | nullIdx ~= floor(nullIdx))
        error('PGIRmanuscript:BadNullIndex', ...
            'nullIdx must contain 1-based integer indices in 1:N.');
    end

    % The manuscript formulas use only the zero-valued null-tone prior.
    if ~isempty(o.pilotIdx) || ~isempty(o.pilotVal)
        error('PGIRmanuscript:NonzeroPilotNotSupported', ...
            ['The manuscript K/E closed forms assume zero-mean Gaussian ', ...
             'time samples and null-tone prior only. Use empty pilotIdx/pilotVal.']);
    end

    scale = sqrt(Px);
    if isempty(o.Tmin),       o.Tmin       = 0.1 * scale; end
    if isempty(o.Tmax),       o.Tmax       = 8.0 * scale; end
    if isempty(o.targetStep), o.targetStep = 0.1 * scale; end
    validateattributes(o.Tmin, {'numeric'}, {'scalar','real','finite','>=',0});
    validateattributes(o.Tmax, {'numeric'}, {'scalar','real','finite','positive'});
    validateattributes(o.targetStep, {'numeric'}, {'scalar','real','finite','positive'});
    if o.Tmax < o.Tmin
        error('PGIRmanuscript:BadThresholdRange', ...
            'Tmax must be greater than or equal to Tmin.');
    end

    % A = F^H I_T'' F: transform-domain projection onto non-null tones.
    F = fft(eye(N)) / sqrt(N);
    nonnullMask = true(N,1);
    nonnullMask(nullIdx) = false;
    A = F' * (nonnullMask .* F);

    % Common random-number pools: one pool is reused for all thresholds.
    knownMask = ~nonnullMask;
    knownFreq = zeros(N,1);                 % null-tone values are zero
    pools = cell(1, o.nSeeds);
    for s = 1:o.nSeeds
        pools{s} = make_pool(N, o.M, o.seed + s - 1, Px, Pw, Pg, p, ...
                             knownMask, knownFreq, F, o.OSFactor);
    end

    Tgrid = threshold_grid(o.Tmin, o.Tmax, o.targetStep);
    nT = numel(Tgrid);
    trace = repmat(empty_trace_row(), nT, 1);
    perSeedGamma_dB = nan(o.nSeeds, nT);

    % Analytic quantities shared by every Monte-Carlo seed.
    Omega0 = Px + Pw;
    Omega1 = Px + Pw + Pg;

    for k = 1:nT
        T = Tgrid(k);
        e0 = exp(-T^2 / Omega0);
        e1 = exp(-T^2 / Omega1);
        pC_noImpulse = (1-p)*e0;          % P(C, Ibar)
        pC_impulse = p*e1;                  % P(C, I)
        pFlag = pC_noImpulse + pC_impulse;

        % Manuscript blanking terms: equations (27) and (33).
        Kblank = 1 - (1-p)*(1 + T^2/Omega0)*e0 ...
                   - p*(1 + T^2/Omega1)*e1;
        Eblank = (1-p)*(Omega0 - (T^2 + Omega0)*e0) ...
                 + p*(Omega1 - (T^2 + Omega1)*e1);

        % Manuscript detected-signal recovery contribution: equation (30).
        RxDetected = (1-p)*Px*(1 + Px*T^2/Omega0^2)*e0 ...
                   + p*Px*(1 + Px*T^2/Omega1^2)*e1;

        totalErr = empty_error_stats();
        for s = 1:o.nSeeds
            recOpt = struct('pool', pools{s}, 'A', A, 'knownTD', zeros(N,1), ...
                            'initialization', o.initialization, ...
                            'fallbackIter', o.steadyStateFallbackIter, ...
                            'rcondTol', o.rcondTol, ...
                            'verifyDirectMC', o.verifyDirectMC);
            [powerE_s, errorSigCon_s, diag_s] = reconErr( ...
                N, nullIdx, T, p, Px, Pw, Pg, o.nIter, o.M, [], [], recOpt);
            totalErr = add_error_stats(totalErr, diag_s);

            % reconErr directly supplies E{e x^*|C} and E{|e|^2|C}.
            % Multiplication by P(C) is exactly the two-event sum in
            % manuscript equations (29) and (34).
            DeltaK_s = pFlag * errorSigCon_s;
            DeltaE_s = pFlag * powerE_s + 2*real(DeltaK_s);
            K_s = Kblank + (RxDetected + DeltaK_s)/Px;
            E_s = Eblank + RxDetected + DeltaE_s;
            perSeedGamma_dB(s,k) = snr_from_K_E(Px, K_s, E_s);
        end

        [powerE, errorSigCon] = conditional_error_moments(totalErr);
        state = conditional_error_moments_by_state(totalErr);
        % DeltaK = P(C)E{e x^*|C}; this equals the explicit manuscript sum
        % over (C,Ibar) and (C,I). The state-wise terms are retained below
        % as a Monte-Carlo diagnostic of that identity.
        DeltaK = pFlag * errorSigCon;
        DeltaE = pFlag * powerE + 2*real(DeltaK);
        DeltaK_by_state = pC_noImpulse*state.errorSigCon_noImpulse ...
                        + pC_impulse*state.errorSigCon_impulse;
        DeltaE_by_state = pC_noImpulse*state.powerE_noImpulse ...
                        + pC_impulse*state.powerE_impulse ...
                        + 2*real(DeltaK_by_state);
        Kpgir = Kblank + (RxDetected + DeltaK)/Px;
        Epgir = Eblank + RxDetected + DeltaE;
        gamma_dB = snr_from_K_E(Px, Kpgir, Epgir);

        tr = empty_trace_row();
        tr.T = T;
        tr.Pflag_theory = pFlag;
        tr.KBlank = Kblank;
        tr.EBlank = Eblank;
        tr.RxDetected = RxDetected;
        tr.powerE_conditional = powerE;
        tr.errorSigCon_conditional = errorSigCon;
        tr.powerE_CIbar = state.powerE_noImpulse;
        tr.errorSigCon_CIbar = state.errorSigCon_noImpulse;
        tr.powerE_CI = state.powerE_impulse;
        tr.errorSigCon_CI = state.errorSigCon_impulse;
        tr.P_CIbar = pC_noImpulse;
        tr.P_CI = pC_impulse;
        tr.DeltaK_by_state = DeltaK_by_state;
        tr.DeltaE_by_state = DeltaE_by_state;
        tr.DeltaK_PGIR = DeltaK;
        tr.DeltaE_PGIR = DeltaE;
        tr.K_PGIR = Kpgir;
        tr.E_PGIR = Epgir;
        tr.gamma_dB = gamma_dB;
        tr.gamma_lin = db2lin_safe(gamma_dB);
        tr.gamma_per_seed_dB = perSeedGamma_dB(:,k).';
        tr.gamma_std_per_seed_dB = finite_std(perSeedGamma_dB(:,k));
        tr = attach_mc_diagnostics(tr, totalErr, Px);
        trace(k) = tr;
    end

    gamma_dB_grid = [trace.gamma_dB];
    finiteIdx = find(isfinite(gamma_dB_grid));
    if isempty(finiteIdx)
        Topt = NaN;
        gammaOpt_dB = NaN;
        bestIdx = NaN;
        best = empty_trace_row();
    else
        [gammaOpt_dB, localIdx] = max(gamma_dB_grid(finiteIdx));
        bestIdx = finiteIdx(localIdx);
        Topt = Tgrid(bestIdx);
        best = trace(bestIdx);
    end

    info = best;
    info.Topt = Topt;
    info.gammaOpt_dB = gammaOpt_dB;
    info.bestIdx = bestIdx;
    info.Tgrid = Tgrid;
    info.gamma_dB_grid = gamma_dB_grid;
    info.gamma_lin_grid = [trace.gamma_lin];
    info.trace = trace;
    info.nIter = o.nIter;
    info.initialization = o.initialization;
    info.M_per_seed = o.M;
    info.OSFactor = o.OSFactor;
    info.nSeeds = o.nSeeds;
    info.M_total = o.M * o.nSeeds;
    info.Knull = numel(nullIdx);
    info.K = best.K_PGIR;                  % compatibility alias
    info.Ey = best.E_PGIR;                 % compatibility alias
    info.DeltaK = best.DeltaK_PGIR;
    info.DeltaE = best.DeltaE_PGIR;
    info.gamma_noproc_dB = 10*log10(Px / (Pw + p*Pg));
    info.convention = ['full second moment; manuscript K/E decomposition; ', ...
                       'DeltaK and DeltaE from reconErr'];
    info.all_frames_retained = true;
    info.manuscript_formula_valid_for = ...
        'null-tone prior, zero-mean circular Gaussian active subcarriers';
end

% ========================================================================
% reconErr is intentionally retained as the error-moment engine.
%
% It returns exactly the two conditional quantities needed by the manuscript:
%
%   powerE      = E{|e_n|^2 | C},
%   errorSigCon = E{ e_n x_n^* | C},
%
% where C is the event |r_n| > T and e_n = xhat_n - x_n at flagged samples.
% The first two outputs preserve the theorem_suboptimal_PGIR.m interface.
%
% No realization is silently discarded. For nIter=Inf, an ill-conditioned
% fixed-point system uses finite-iteration PGIR and is counted in diagnostics.
% ========================================================================
function [powerE, errorSigCon, st] = reconErr(fftSize, nullMask, T, p, ...
        powerS, powerW, powerG, iterCount, Msample, pilotIdx, pilotVal, varargin)
%RECONERR  Estimate the two conditional reconstruction-error moments.
%
% This keeps the original theorem_suboptimal_PGIR.m interface:
%
%   [powerE,errorSigCon] = reconErr(fftSize,nullMask,T,p,powerS,powerW, ...
%       powerG,iterCount,Msample,pilotIdx,pilotVal)
%
% and returns
%   powerE      = E{|e_n|^2 | |r_n|>T},
%   errorSigCon = E{e_n x_n^* | |r_n|>T}.
%
% Optional twelfth input: a struct with fields pool, A, knownTD,
% initialization, fallbackIter, rcondTol, and verifyDirectMC.  The main
% optimizer supplies this struct to reuse common random numbers.  Without
% it, reconErr remains callable exactly as in the original theorem script.

    if nargin < 10 || isempty(pilotIdx), pilotIdx = []; end
    if nargin < 11 || isempty(pilotVal), pilotVal = []; end

    opt = struct('pool', [], 'A', [], 'knownTD', [], ...
                 'initialization', 'zero', 'fallbackIter', 50, ...
                 'rcondTol', 1e-10, 'verifyDirectMC', false, ...
                 'OSFactor', 1);
    if ~isempty(varargin)
        userOpt = varargin{1};
        if ~isstruct(userOpt)
            error('PGIRmanuscript:BadReconErrOption', ...
                'Optional reconErr input must be a struct.');
        end
        f = fieldnames(userOpt);
        for k = 1:numel(f)
            if isfield(opt, f{k})
                opt.(f{k}) = userOpt.(f{k});
            end
        end
    end

    [knownMask, knownFreq] = make_known_constraint(fftSize, nullMask, ...
                                                     pilotIdx, pilotVal);
    if isempty(opt.A)
        F = fft(eye(fftSize))/sqrt(fftSize);
        opt.A = F' * ((~knownMask) .* F);
    end
    if isempty(opt.knownTD)
        F = fft(eye(fftSize))/sqrt(fftSize);
        opt.knownTD = F' * (knownMask .* knownFreq);
    end
    if isempty(opt.pool)
        F = fft(eye(fftSize))/sqrt(fftSize);
        opt.pool = make_pool(fftSize, Msample, 0, powerS, powerW, powerG, p, ...
                             knownMask, knownFreq, F, opt.OSFactor);
    end

    st = reconErr_pool(opt.pool, T, opt.A, opt.knownTD, iterCount, ...
                       opt.initialization, opt.fallbackIter, opt.rcondTol, ...
                       opt.verifyDirectMC);
    [powerE, errorSigCon] = conditional_error_moments(st);
end

function st = reconErr_pool(pool, T, A, knownTD, nIter, initialization, ...
                            fallbackIter, rcondTol, verifyDirectMC)
    st = empty_error_stats();

    for m = 1:pool.M
        x = pool.x(:,m);
        u = pool.u(:,m);
        b = pool.b(:,m);
        r = x + u;
        flag = abs(r) > T;
        U = nnz(flag);

        [xhat, rec] = pgir_reconstruct(r, flag, A, knownTD, nIter, ...
                                       initialization, fallbackIter, rcondTol);
        y = r;
        y(flag) = xhat(flag);

        st.Nframes = st.Nframes + 1;
        st.Nsamples = st.Nsamples + numel(x);
        st.Nflag = st.Nflag + U;
        st.NflagFrames = st.NflagFrames + double(U > 0);
        st.Nfallback = st.Nfallback + double(rec.usedFallback);
        st.Nillconditioned = st.Nillconditioned + double(rec.wasIllConditioned);

        if U > 0
            e = xhat(flag) - x(flag);
            st.sumErr2 = st.sumErr2 + sum(abs(e).^2);
            st.sumErrX = st.sumErrX + sum(e .* conj(x(flag)));
            st.NflagForMoments = st.NflagForMoments + U;

            flagNoImpulse = flag & ~b;
            flagImpulse = flag & b;
            if any(flagNoImpulse)
                e0 = xhat(flagNoImpulse) - x(flagNoImpulse);
                st.sumErr2NoImpulse = st.sumErr2NoImpulse + sum(abs(e0).^2);
                st.sumErrXNoImpulse = st.sumErrXNoImpulse + sum(e0 .* conj(x(flagNoImpulse)));
                st.NflagNoImpulse = st.NflagNoImpulse + nnz(flagNoImpulse);
            end
            if any(flagImpulse)
                e1 = xhat(flagImpulse) - x(flagImpulse);
                st.sumErr2Impulse = st.sumErr2Impulse + sum(abs(e1).^2);
                st.sumErrXImpulse = st.sumErrXImpulse + sum(e1 .* conj(x(flagImpulse)));
                st.NflagImpulse = st.NflagImpulse + nnz(flagImpulse);
            end
        end

        if verifyDirectMC
            st.sumYX = st.sumYX + x' * y;
            st.sumY2 = st.sumY2 + real(y' * y);
            st.sumX2 = st.sumX2 + real(x' * x);
        end

        if isfinite(rec.rcondValue)
            st.rcondValues(end+1,1) = rec.rcondValue; %#ok<AGROW>
        end
    end
end

% ========================================================================
% Physically realizable PGIR reconstruction. The detector has already set
% the data-domain reliable mask. A is F^H I_T' F and knownTD is F^H I_T X.
% The manuscript optimizer invokes this routine with null-only knownTD=0;
% reconErr itself also retains optional known-pilot support.
% ========================================================================
function [xhat, rec] = pgir_reconstruct(r, flag, A, knownTD, nIter, initialization, ...
                                         fallbackIter, rcondTol)
    N = numel(r);
    xhat = zeros(N,1);
    rec = struct('usedFallback', false, 'wasIllConditioned', false, ...
                 'rcondValue', NaN);

    if strcmp(initialization, 'data')
        xhat(~flag) = r(~flag);             % xhat^(0) = I_D r
    end

    U = nnz(flag);
    if U == 0
        xhat = r;                           % P_D always pins all samples
        return;
    end

    reli = ~flag;
    Aff = A(flag, flag);
    Afr = A(flag, reli);
    knownFlag = knownTD(flag);
    rhs = knownFlag + Afr * r(reli);

    if isinf(nIter)
        R = eye(U) - Aff;
        rc = rcond(R);
        rec.rcondValue = rc;
        if isfinite(rc) && rc >= rcondTol
            xhatFlag = R \ rhs;
        else
            rec.usedFallback = true;
            rec.wasIllConditioned = true;
            xhatFlag = finite_pgir_flagged(Aff, rhs, knownFlag, initialization, fallbackIter);
        end
    else
        xhatFlag = finite_pgir_flagged(Aff, rhs, knownFlag, initialization, nIter);
    end

    xhat(reli) = r(reli);
    xhat(flag) = xhatFlag;
    assert(numel(xhat) == N);
end

function xhatFlag = finite_pgir_flagged(Aff, rhs, knownFlag, initialization, L)
%FINITE_PGIR_FLAGGED  Reduced exact recursion of xhat^(i+1)=P_D(P_T(xhat^i)).
%
% For initialization='data', xhat^(0)=I_D r and the flagged recursion is
%       xhat_F^(i+1) = A_FF xhat_F^(i) + A_FR r_R.
%
% For initialization='zero', the first projection starts from xhat^(0)=0,
% so xhat_F^(1)=knownFlag; reliable samples become r_R after that first
% P_D operation, and the above recurrence applies from i=1.

    U = size(Aff,1);
    xhatFlag = zeros(U,1);
    if L == 0
        return;
    end

    switch initialization
        case 'data'
            for it = 1:L
                xhatFlag = Aff*xhatFlag + rhs;
            end
        case 'zero'
            % Iteration 1: P_D(P_T(0)) = known transform-domain prior.
            xhatFlag = knownFlag;
            for it = 2:L
                xhatFlag = Aff*xhatFlag + rhs;
            end
        otherwise
            error('PGIRmanuscript:InternalInitialization', ...
                'Unknown initialization mode.');
    end
end

% ========================================================================
% Generate Gaussian active-subcarrier OFDM samples. OSFactor is applied
% only to the OFDM normalization, matching
%
%   x = sqrt(OSFactor) F^H X.
%
% The active-subcarrier variance is reduced by the same factor so that
% every x_n still has the requested full second moment Px. For
% OSFactor = 1, this reduces exactly to the original implementation.
% ========================================================================
function pool = make_pool(N, M, seed, Px, Pw, Pg, p, knownMask, knownFreq, F, OSFactor)
    rng(seed, 'twister');
    randomIdx = find(~knownMask);
    if isempty(randomIdx)
        error('PGIRmanuscript:NoDataTone', ...
            'At least one transform-domain data subcarrier is required.');
    end

    if nargin < 11 || isempty(OSFactor)
        OSFactor = 1;
    end
    validateattributes(OSFactor, {'numeric'}, ...
        {'scalar','real','finite','positive'});

    knownEnergy = sum(abs(knownFreq(knownMask)).^2);
    targetFreqEnergy = N*Px/OSFactor;
    Psub = (targetFreqEnergy - knownEnergy) / numel(randomIdx);
    if Psub < -100*eps(max(targetFreqEnergy,1))
        error('PGIRmanuscript:PowerBudget', ...
            'Known pilot energy exceeds the requested time-domain power Px.');
    end
    Psub = max(Psub, 0);

    X = repmat(knownFreq, 1, M);
    X(randomIdx,:) = (randn(numel(randomIdx),M) + 1i*randn(numel(randomIdx),M)) ...
                     * sqrt(Psub/2);
    pool.x = sqrt(OSFactor) * (F' * X);

    W = (randn(N,M) + 1i*randn(N,M)) * sqrt(Pw/2);
    G = (randn(N,M) + 1i*randn(N,M)) * sqrt(Pg/2);
    B = rand(N,M) < p;
    pool.u = W + B .* G;
    pool.b = B;
    pool.M = M;
end

function [knownMask, knownFreq] = make_known_constraint(N, nullIdx, pilotIdx, pilotVal)
    nullIdx = unique(nullIdx(:));
    if any(nullIdx < 1 | nullIdx > N | nullIdx ~= floor(nullIdx))
        error('PGIRmanuscript:BadNullIndex', ...
            'nullIdx must contain 1-based integer indices in 1:N.');
    end

    pilotIdx = unique(pilotIdx(:));
    if any(pilotIdx < 1 | pilotIdx > N | pilotIdx ~= floor(pilotIdx))
        error('PGIRmanuscript:BadPilotIndex', ...
            'pilotIdx must contain 1-based integer indices in 1:N.');
    end
    if any(ismember(pilotIdx, nullIdx))
        error('PGIRmanuscript:OverlappingKnownTone', ...
            'pilotIdx and nullIdx must not overlap.');
    end

    if isempty(pilotIdx)
        pilotVal = zeros(0,1);
    else
        if isempty(pilotVal)
            error('PGIRmanuscript:MissingPilotValue', ...
                'pilotVal is required for a nonempty pilotIdx.');
        end
        pilotVal = pilotVal(:);
        if isscalar(pilotVal)
            pilotVal = repmat(pilotVal, numel(pilotIdx), 1);
        elseif numel(pilotVal) ~= numel(pilotIdx)
            error('PGIRmanuscript:BadPilotValue', ...
                'pilotVal must be scalar or match numel(pilotIdx).');
        end
    end

    knownMask = false(N,1);
    knownMask(nullIdx) = true;
    knownMask(pilotIdx) = true;
    knownFreq = zeros(N,1);
    knownFreq(pilotIdx) = pilotVal;
end

function [powerE, errorSigCon] = conditional_error_moments(st)
    if st.NflagForMoments == 0
        powerE = 0;
        errorSigCon = 0;
    else
        powerE = real(st.sumErr2 / st.NflagForMoments);
        errorSigCon = st.sumErrX / st.NflagForMoments;
    end
end

function state = conditional_error_moments_by_state(st)
%CONDITIONAL_ERROR_MOMENTS_BY_STATE  The two terms appearing in manuscript
% (29) and (34): conditional on (C,Ibar) and on (C,I), respectively.
    state = struct('powerE_noImpulse',0, 'errorSigCon_noImpulse',0, ...
                   'powerE_impulse',0, 'errorSigCon_impulse',0);
    if st.NflagNoImpulse > 0
        state.powerE_noImpulse = real(st.sumErr2NoImpulse / st.NflagNoImpulse);
        state.errorSigCon_noImpulse = st.sumErrXNoImpulse / st.NflagNoImpulse;
    end
    if st.NflagImpulse > 0
        state.powerE_impulse = real(st.sumErr2Impulse / st.NflagImpulse);
        state.errorSigCon_impulse = st.sumErrXImpulse / st.NflagImpulse;
    end
end

function st = empty_error_stats()
    st = struct('sumErr2',0, 'sumErrX',0, 'NflagForMoments',0, ...
                'sumErr2NoImpulse',0, 'sumErrXNoImpulse',0, 'NflagNoImpulse',0, ...
                'sumErr2Impulse',0, 'sumErrXImpulse',0, 'NflagImpulse',0, ...
                'Nsamples',0, 'Nframes',0, 'Nflag',0, 'NflagFrames',0, ...
                'Nfallback',0, 'Nillconditioned',0, ...
                'sumYX',0, 'sumY2',0, 'sumX2',0, ...
                'rcondValues',zeros(0,1));
end

function out = add_error_stats(a, b)
    out = a;
    f = {'sumErr2','sumErrX','NflagForMoments', ...
         'sumErr2NoImpulse','sumErrXNoImpulse','NflagNoImpulse', ...
         'sumErr2Impulse','sumErrXImpulse','NflagImpulse', ...
         'Nsamples','Nframes','Nflag','NflagFrames','Nfallback','Nillconditioned', ...
         'sumYX','sumY2','sumX2'};
    for k = 1:numel(f)
        out.(f{k}) = a.(f{k}) + b.(f{k});
    end
    out.rcondValues = [a.rcondValues; b.rcondValues];
end

function tr = empty_trace_row()
    tr = struct('T',NaN, 'Pflag_theory',NaN, ...
        'KBlank',NaN, 'EBlank',NaN, 'RxDetected',NaN, ...
        'powerE_conditional',NaN, 'errorSigCon_conditional',NaN, ...
        'powerE_CIbar',NaN, 'errorSigCon_CIbar',NaN, ...
        'powerE_CI',NaN, 'errorSigCon_CI',NaN, ...
        'P_CIbar',NaN, 'P_CI',NaN, ...
        'DeltaK_by_state',NaN, 'DeltaE_by_state',NaN, ...
        'DeltaK_PGIR',NaN, 'DeltaE_PGIR',NaN, ...
        'K_PGIR',NaN, 'E_PGIR',NaN, 'gamma_lin',NaN, 'gamma_dB',NaN, ...
        'gamma_per_seed_dB',[], 'gamma_std_per_seed_dB',NaN, ...
        'Pflag_emp',NaN, 'U_mean',NaN, 'P_flagged_frame_emp',NaN, ...
        'P_fallback_emp',NaN, 'P_illconditioned_emp',NaN, ...
        'rcond_min',NaN, 'rcond_01',NaN, 'cond_max',NaN, 'cond99',NaN, ...
        'K_direct_MC',NaN, 'E_direct_MC',NaN, ...
        'gamma_direct_MC_dB',NaN, 'K_formula_minus_direct',NaN, ...
        'E_formula_minus_direct',NaN);
end

function tr = attach_mc_diagnostics(tr, st, Px)
    if st.Nsamples == 0
        return;
    end

    tr.Pflag_emp = st.Nflag / st.Nsamples;
    tr.U_mean = st.Nflag / st.Nframes;
    tr.P_flagged_frame_emp = st.NflagFrames / st.Nframes;
    tr.P_fallback_emp = st.Nfallback / st.Nframes;
    tr.P_illconditioned_emp = st.Nillconditioned / st.Nframes;

    if ~isempty(st.rcondValues)
        tr.rcond_min = min(st.rcondValues);
        tr.rcond_01 = percentile(st.rcondValues, 1);
        tr.cond_max = 1/max(tr.rcond_min, realmin);
        tr.cond99 = 1/max(tr.rcond_01, realmin);
    end

    if st.sumX2 > 0
        tr.K_direct_MC = st.sumYX / st.sumX2;
        tr.E_direct_MC = real(st.sumY2 / st.Nsamples);
        tr.gamma_direct_MC_dB = snr_from_K_E(Px, tr.K_direct_MC, tr.E_direct_MC);
        tr.K_formula_minus_direct = tr.K_PGIR - tr.K_direct_MC;
        tr.E_formula_minus_direct = tr.E_PGIR - tr.E_direct_MC;
    end
end

function gamma_dB = snr_from_K_E(Px, K, Eout)
    signalPower = Px * abs(K)^2;
    distortionPower = real(Eout - signalPower);
    if isfinite(signalPower) && isfinite(distortionPower) && ...
            signalPower > 0 && distortionPower > 0
        gamma_dB = 10*log10(signalPower / distortionPower);
    else
        gamma_dB = NaN;
    end
end

function y = db2lin_safe(x)
    if isfinite(x)
        y = 10^(x/10);
    else
        y = NaN;
    end
end

function grid = threshold_grid(Tmin, Tmax, step)
    n = floor((Tmax - Tmin)/step);
    grid = Tmin + (0:n)*step;
    tol = 32*eps(max(1,Tmax));
    if isempty(grid) || abs(grid(end) - Tmax) > tol
        grid(end+1) = Tmax;
    end
    grid = unique(grid, 'stable');
end

function s = finite_std(x)
    x = x(isfinite(x));
    if numel(x) <= 1
        s = 0;
    else
        s = std(x,0);
    end
end

function qv = percentile(x, q)
    if isempty(x)
        qv = NaN;
        return;
    end
    x = sort(x(:));
    n = numel(x);
    if n == 1
        qv = x;
        return;
    end
    pos = 1 + (n-1)*(q/100);
    lo = floor(pos);
    hi = ceil(pos);
    qv = x(lo) + (pos-lo)*(x(hi)-x(lo));
end
