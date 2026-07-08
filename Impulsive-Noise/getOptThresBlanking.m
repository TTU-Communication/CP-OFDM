function [Topt, gamma_max, info] = getOptThresBlanking(Px, Pw, Pg, p)
%OPTIMALTHRESHOLD_MANUSCRIPT  Blanking 最佳門檻 (manuscript 能量定義)
%
%   採用 manuscript 的「全二階矩 (full second-moment)」正規化慣例 (見 manuscript
%   "Normalization convention" 段落)：
%       sigma_x^2 = E{|x_n|^2}
%       sigma_w^2 = E{|w_n|^2}
%       sigma_g^2 = E{|g_n|^2}
%   |r_n| 的 Rayleigh 參數在無脈衝/有脈衝時為 (sigma_x^2+sigma_w^2) 與
%   (sigma_x^2+sigma_w^2+sigma_g^2)，指數項呈 exp(-T^2/(sigma_x^2+sigma_w^2))
%   形式 (而非 Zhidkov 的 exp(-T^2/(2(1+sigma_w^2))))。
%
%   注意：此處計算的是 manuscript 中與 blanking 接收機重合的 reliable/clipping
%   事件貢獻，即 K_Blank (式 27) 與 E_Blank (式 33) 代入 output SNR (式 22)。
%   這是「只用純量能量」即可得到的封閉式部分；完整 PGIR 之 gamma 還會加入
%   recovery 項 (式 31、35 中的 Delta K^{(inf)}_PGIR、Delta E^{(inf)}_PGIR)，
%   其需 null/unreliable 幾何 (S 矩陣、N、null 數) 才能算出，無法僅由純量能量決定。
%
%   輸入 (皆為 manuscript 慣例下的能量, 即全二階矩):
%       Px : 訊號能量 sigma_x^2
%       Pw : 背景 AWGN 能量 sigma_w^2
%       Pg : 脈衝雜訊 Gaussian 分量能量 sigma_g^2
%       p  : 脈衝出現機率 Pr(b_n = 1)
%
%   輸出:
%       Topt      : 使 output SNR (式 22) 最大的最佳門檻 T
%       gamma_max : 對應的最大 output SNR (線性值, 非 dB)
%       info      : 結構, 含 gamma_dB 等
%
%   公式對應 manuscript 式號:
%       K_Blank -> 式 (27)
%       E_Blank -> 式 (33)
%       gamma   -> 式 (22)

    if nargin < 4
        error('需要 4 個輸入: Px, Pw, Pg, p');
    end

    Tmax = 8;

    Tgrid = linspace(1e-6, Tmax, 8000);
    g     = arrayfun(@(T) ms_gamma(T, Px, Pw, Pg, p), Tgrid);
    [~, idx] = max(g);
    lo = Tgrid(max(idx-2,1));
    hi = Tgrid(min(idx+2,numel(Tgrid)));
    opts = optimset('TolX',1e-10);
    [Topt, negg] = fminbnd(@(T) -ms_gamma(T, Px, Pw, Pg, p), lo, hi, opts);
    gamma_max = -negg;

    % 交叉驗證: 解 dln E_Blank/dT - 2 dln K_Blank/dT = 0  (同 Zhidkov 式 18 結構)
    dObj = @(T) dlogE(T,Px,Pw,Pg,p) - 2*dlogK(T,Px,Pw,Pg,p);
    Troot = NaN;
    try
        Troot = fzero(dObj, Topt);
    catch
    end

    info = struct();
    info.gamma_max_dB = 10*log10(gamma_max);
    info.Topt_eq18    = Troot;
    info.K_at_opt     = ms_KBlank(Topt,Px,Pw,Pg,p);
    info.E_at_opt     = ms_EBlank(Topt,Px,Pw,Pg,p);
    info.gamma_Tinf   = Px/(Pw + p*Pg);   % T->inf 之傳統接收機極限 (全矩慣例)
    info.convention   = 'manuscript (full second-moment, exp(-T^2/(Px+Pw)))';
    info.note         = '此為 K_Blank/E_Blank 之 blanking baseline; 完整 PGIR 需加 Delta E 幾何項';
end

% ===================== 區域封閉式函式 =====================
function K = ms_KBlank(T, Px, Pw, Pg, p)         % 式 (27)
    b0 = Px + Pw;
    b1 = Px + Pw + Pg;
    K = 1 ...
        - (1-p).*(1 + T.^2./b0).*exp(-T.^2./b0) ...
        -    p .*(1 + T.^2./b1).*exp(-T.^2./b1);
end

function E = ms_EBlank(T, Px, Pw, Pg, p)         % 式 (33)
    b0 = Px + Pw;
    b1 = Px + Pw + Pg;
    E = (1-p).*( b0 - (T.^2 + b0).*exp(-T.^2./b0) ) ...
      +    p .*( b1 - (T.^2 + b1).*exp(-T.^2./b1) );
end

function g = ms_gamma(T, Px, Pw, Pg, p)          % 式 (22): gamma = (E|y|^2/(Px K^2) - 1)^-1
    K = ms_KBlank(T, Px, Pw, Pg, p);
    E = ms_EBlank(T, Px, Pw, Pg, p);
    g = 1 ./ (E ./ (Px.*K.^2) - 1);
end

function d = dlogK(T, Px, Pw, Pg, p)
    h = 1e-6*max(T,1);
    d = (log(ms_KBlank(T+h,Px,Pw,Pg,p)) - log(ms_KBlank(T-h,Px,Pw,Pg,p)))/(2*h);
end
function d = dlogE(T, Px, Pw, Pg, p)
    h = 1e-6*max(T,1);
    d = (log(ms_EBlank(T+h,Px,Pw,Pg,p)) - log(ms_EBlank(T-h,Px,Pw,Pg,p)))/(2*h);
end
