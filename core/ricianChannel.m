function [fadedSig, H] = ricianChannel(inSig, nTap, kFactor, fftSize, nRX)

    arguments
        inSig (:,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty}
        nTap (1,1) double {mustBePositive, mustBeInteger}
        kFactor (1,1) double {mustBeNonnegative}
        fftSize (1,1) double {mustBePositive, mustBeInteger}
        nRX (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    [sigSample, sigBatch, nTX] = size(inSig);

    losPower = kFactor / (kFactor + 1);
    nlosPower = 1 / (kFactor + 1) * 1 / nTap;
    hLOS = sqrt(losPower) * exp(1j * 2 * pi * rand) + ...
        sqrt(nlosPower) * randn([1, sigBatch, nRX, nTX], 'like', inSig(1));
    hNLOS = sqrt(nlosPower) .* randn([nTap-1, sigBatch, nRX, nTX], 'like', inSig(1));
    h = cat(1, hLOS, hNLOS);
    convLen = sigSample + nTap - 1;

    % calculate fft of INSIG at first dimension
    inSigF = fft([inSig;  zeros([convLen - sigSample, sigBatch, nTX], 'like', inSig)], convLen, 1);
    % change dimension from [Nsp Ns Ntx] to [Ntx 1 Nsp Ns]
    inSigFPage = reshape(permute(inSigF, [3 1 2]), [nTX 1 convLen sigBatch]);
    % calculate fft of h at first dimension
    hF = fft([h; zeros([convLen - nTap, sigBatch, nRX, nTX], 'like', inSig)], convLen, 1);
    % change dimension from [Nsp Ns Nrx Ntx] to [Nrx Ntx Nsp Ns]
    hFPage = permute(hF, [3 4 1 2]);

    % calculate convolution
    yFPage = pagemtimes(hFPage, inSigFPage);
    % change dimension from [Nrx 1 Nsp Ns] to [Nsp Ns Nrx]
    yF = permute(reshape(yFPage, [nRX convLen sigBatch]), [2 3 1]);
    % calculate ifft of YF at first dimension
    y = ifft(yF, convLen, 1);

    fadedSig = y(1:sigSample, :, :);
    H = fft(cat(1, h, zeros([fftSize - nTap, sigBatch, nRX, nTX])), fftSize, 1);

end
