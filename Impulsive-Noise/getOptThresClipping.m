function [Topt, gamma_max, info] = getOptThresClipping(Px, Pw, Pg, p)
%OPTIMALTHRESHOLDCLIP_MANUSCRIPT  Clipping 最佳門檻 (manuscript 能量定義)
%
%   Clipping 非線性 (同 Zhidkov 2008 式 7):
%       y_n = r_n,                 |r_n| <= T
%           = T * exp(j arg(r_n)),  |r_n| >  T
%
%   採用 manuscript 的「全二階矩」正規化慣例 (見 manuscript "Normalization
%   convention" 段落), 與 blanking 版本一致:
%       sigma_x^2 = E{|x_n|^2},  sigma_w^2 = E{|w_n|^2},  sigma_g^2 = E{|g_n|^2}
%   每事件接收功率 (= Rayleigh 二階矩):
%       Omega_0 = sigma_x^2 + sigma_w^2              (無脈衝)
%       Omega_1 = sigma_x^2 + sigma_w^2 + sigma_g^2  (有脈衝)
%   指數項呈 exp(-T^2/Omega_l) (無 Zhidkov 的因子 2)。
%
%   注意: Zhidkov 2008 的 clipping 推導本以半矩慣例寫成; 本檔將其改寫為
%   manuscript 的全矩慣例 (sigma_x^2 取代 Zhidkov 的 2 sigma_s^2)。輸出 SNR
%   採 manuscript 式 (22) 的形式 gamma = (Eout/(sigma_x^2 K0^2) - 1)^-1。
%   (clipping 為純振幅非線性, 不含 PGIR 重建; 故無 Delta E 幾何項。)
%
%   輸入 (manuscript 慣例, 全二階矩):
%       Px : 訊號能量 sigma_x^2
%       Pw : AWGN 能量 sigma_w^2
%       Pg : 脈衝雜訊能量 sigma_g^2
%       p  : 脈衝出現機率
%
%   輸出:
%       Topt, gamma_max(線性), info(含 gamma_dB 等)

    if nargin < 4, error('需要 4 個輸入: Px, Pw, Pg, p'); end

    Om0 = Px + Pw;
    Om1 = Px + Pw + Pg;
    Esig = Px;                            % E{|x_n|^2} = sigma_x^2

    Tmax = 8*sqrt(Om1);
    Tgrid = linspace(1e-6, Tmax, 8000);
    g = arrayfun(@(T) clip_gamma(T,Om0,Om1,p,Esig), Tgrid);
    [~,idx] = max(g);
    lo = Tgrid(max(idx-2,1)); hi = Tgrid(min(idx+2,numel(Tgrid)));
    [Topt, negg] = fminbnd(@(T) -clip_gamma(T,Om0,Om1,p,Esig), lo, hi, optimset('TolX',1e-10));
    gamma_max = -negg;

    dObj = @(T) dlog(@clip_E,T,Om0,Om1,p,Esig) - 2*dlog(@clip_K0,T,Om0,Om1,p,Esig);
    Troot = NaN; try, Troot = fzero(dObj, Topt); catch, end

    info = struct();
    info.gamma_max_dB = 10*log10(gamma_max);
    info.Topt_eq_opt  = Troot;
    info.K0_at_opt    = clip_K0(Topt,Om0,Om1,p,Esig);
    info.Eout_at_opt  = clip_E(Topt,Om0,Om1,p,Esig);
    info.gamma_Tinf   = Px/(Pw + p*Pg);
    info.convention   = 'manuscript (full second-moment, Omega_l = Px+...)';
end

% ===== 區域函式 (Omega_l = 每事件接收功率) =====
function K0 = clip_K0(T, Om0, Om1, p, ~)
    K0 = 1 - (1-p)*term(T,Om0) - p*term(T,Om1);
end
function t = term(T, Om)
    t = exp(-T.^2./Om) - sqrt(pi).*(T./sqrt(Om)).*Qf(sqrt(2).*T./sqrt(Om));
end
function E = clip_E(T, Om0, Om1, p, ~)
    E = (1-p).*Om0.*(1-exp(-T.^2./Om0)) + p.*Om1.*(1-exp(-T.^2./Om1));
end
function g = clip_gamma(T, Om0, Om1, p, Esig)       % 式 (22)
    K0 = clip_K0(T,Om0,Om1,p,Esig);
    E  = clip_E(T,Om0,Om1,p,Esig);
    g  = 1 ./ (E ./ (Esig.*K0.^2) - 1);
end
function q = Qf(x)
    q = 0.5*erfc(x./sqrt(2));
end
function d = dlog(fn, T, Om0, Om1, p, Esig)
    h = 1e-6*max(T,1);
    d = (log(fn(T+h,Om0,Om1,p,Esig)) - log(fn(T-h,Om0,Om1,p,Esig)))/(2*h);
end
