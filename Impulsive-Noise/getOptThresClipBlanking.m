function [Tc_opt, gamma_max, info] = getOptThresClipBlanking(Px, Pw, Pg, p, ratio)
%OPTIMALTHRESHOLDCLIPBLANK_MANUSCRIPT  Clipping+Blanking 最佳門檻 (manuscript 能量定義)
%
%   複合 clipping/blanking 非線性 (同 Zhidkov 2008 式 8):
%       y_n = r_n,                  |r_n| <= T1
%           = T1 * exp(j arg(r_n)),  T1 < |r_n| <= T2
%           = 0,                     |r_n| >  T2
%   T1 = T_c (clipping 門檻), T2 = T_b (blanking 門檻)。
%   本系統設定: T_c = T, T_b = 1.4 T_c  => ratio = 1.4 (預設)。對 T_c 最佳化。
%
%   採用 manuscript 全二階矩慣例 (見 "Normalization convention" 段落):
%       sigma_x^2 = E{|x_n|^2}, sigma_w^2 = E{|w_n|^2}, sigma_g^2 = E{|g_n|^2}
%   每事件接收功率:
%       Omega_0 = sigma_x^2 + sigma_w^2
%       Omega_1 = sigma_x^2 + sigma_w^2 + sigma_g^2
%   指數項呈 exp(-T^2/Omega_l) (無 Zhidkov 因子 2)。輸出 SNR 採 manuscript
%   式 (22) 形式 gamma = (Eout/(sigma_x^2 K0^2) - 1)^-1。
%   (clip/blank 為純振幅非線性, 不含 PGIR 重建, 故無 Delta E 幾何項。)
%
%   輸入 (manuscript 慣例, 全二階矩):
%       Px, Pw, Pg : sigma_x^2, sigma_w^2, sigma_g^2
%       p          : 脈衝出現機率
%       ratio      : T_b / T_c (選用, 預設 1.4)
%
%   輸出:
%       Tc_opt, gamma_max(線性), info(含 Tb_opt、gamma_dB 等)

    if nargin < 4, error('需要至少 4 個輸入: Px, Pw, Pg, p'); end
    if nargin < 5 || isempty(ratio), ratio = 1.4; end
    if ratio < 1, error('ratio = T_b/T_c 必須 >= 1'); end

    Om0 = Px + Pw;
    Om1 = Px + Pw + Pg;
    Esig = Px;                           % E{|x_n|^2} = sigma_x^2

    Tmax = 8*sqrt(Om1)/ratio;
    Tgrid = linspace(1e-6, Tmax, 8000);
    g = arrayfun(@(Tc) cb_gamma(Tc, ratio*Tc, Om0, Om1, p, Esig), Tgrid);
    [~,idx] = max(g);
    lo = Tgrid(max(idx-2,1)); hi = Tgrid(min(idx+2,numel(Tgrid)));
    [Tc_opt, negg] = fminbnd(@(Tc) -cb_gamma(Tc, ratio*Tc, Om0, Om1, p, Esig), ...
                             lo, hi, optimset('TolX',1e-10));
    gamma_max = -negg;

    info = struct();
    info.Tb_opt       = ratio*Tc_opt;
    info.ratio        = ratio;
    info.gamma_max_dB = 10*log10(gamma_max);
    info.K0_at_opt    = cb_K0(Tc_opt, ratio*Tc_opt, Om0, Om1, p);
    info.Eout_at_opt  = cb_E (Tc_opt, ratio*Tc_opt, Om0, Om1, p);
    info.gamma_Tinf   = Px/(Pw + p*Pg);
    info.convention   = 'manuscript (full second-moment, Omega_l = Px+...)';
end

% ===== 區域函式 (Omega_l = 每事件接收功率) =====
function K0 = cb_K0(T1, T2, Om0, Om1, p)
    K0 = 1 - (1-p)*K0t(T1,T2,Om0) - p*K0t(T1,T2,Om1);
end
function v = K0t(T1, T2, Om)
    v = exp(-T1.^2./Om) + (T1.*T2./Om).*exp(-T2.^2./Om) ...
        - sqrt(pi).*(T1./sqrt(Om)).*( Qf(sqrt(2).*T1./sqrt(Om)) - Qf(sqrt(2).*T2./sqrt(Om)) );
end
function E = cb_E(T1, T2, Om0, Om1, p)
    E = (1-p)*Et(T1,T2,Om0) + p*Et(T1,T2,Om1);
end
function v = Et(T1, T2, Om)
    v = Om.*(1 - exp(-T1.^2./Om)) - T1.^2.*exp(-T2.^2./Om);
end
function g = cb_gamma(T1, T2, Om0, Om1, p, Esig)    % 式 (22)
    K0 = cb_K0(T1,T2,Om0,Om1,p);
    E  = cb_E (T1,T2,Om0,Om1,p);
    g  = 1 ./ (E ./ (Esig.*K0.^2) - 1);
end
function q = Qf(x), q = 0.5*erfc(x./sqrt(2)); end
