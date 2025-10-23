function [y, gain] = agc(x, frameLen, targetRMS, maxGain_dB, minGain_dB, mode)
%A Summary of this function goes here
%   Detailed explanation goes here
    arguments (Input)
        x (:,:, :) {mustBeArrayOrGPU, mustBeFinite}
        frameLen (1,1) double {mustBeInteger} = 32
        targetRMS (1,1) double {mustBeFinite, mustBeReal} = 0.25
        maxGain_dB (1,1) double {mustBeFinite, mustBeReal} = 80
        minGain_dB (1,1) double {mustBeFinite, mustBeReal} = 0
        mode char {mustBeMember(mode, {'sum', 'max'})} = 'sum'
    end
    
    sz = size(x);
    x = reshape(x, sz(1) * sz(2), max(1, prod(sz(3:end))));
    [Ns, Nrx] = size(x);
    
    pad = mod(-Ns, frameLen);
    Xp = cat(1, x, zeros(pad, Nrx, 'like', x));
    Nf = size(Xp, 1) / frameLen;
    Xf = reshape(Xp, frameLen, Nf, Nrx);
    P = mean(abs(Xf).^2, 1);

    switch lower(mode)
        case 'sum', Pr = sum(P, 3);
        case 'max', Pr = max(P,[], 3);
    end
    rmsFrame = sqrt(Pr);
    g = targetRMS ./ (rmsFrame);
    g = max(min(g, 10 .^ (maxGain_dB / 20)), 10 .^ (minGain_dB / 20));
    gain = g;

    G = repmat(g, [frameLen 1 Nrx]);
    Yf = G .* Xf;
    Y = reshape(Yf, frameLen*Nf, Nrx);
    y = Y(1:Ns, :, :);
    y = reshape(y, sz);
end