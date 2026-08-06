function [fadedSig, H] = rayleighChannel(inSig, nTap, tapPower, fftSize, nRX)

    arguments
        inSig (:,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty}
        nTap (1,1) double {mustBePositive, mustBeInteger}
        tapPower (1,1) double {mustBeNonnegative, mustBeReal, mustBeFinite}
        fftSize (1,1) double {mustBePositive, mustBeInteger}
        nRX (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    [nSample, nTX, nBatch] = size(inSig);

    h = sqrt(tapPower) * randn([nTap, nRX, nTX, nBatch], 'like', inSig(1));
    convLen = nSample + nTap - 1;

    % calculate fft of INSIG at first dimension
    inSigF = fft(cat(1, inSig, zeros([convLen - nSample, nTX, nBatch], 'like', inSig(1))), convLen, 1);
    % change dimension from [Nsample Ntx Nbatch] to [Ntx 1 Nsample Nbatch]
    inSigFPage = reshape(permute(inSigF, [2 1 3]), [nTX 1 convLen nBatch]);
    % calculate fft of h at first dimension
    hF = fft(cat(1, h, zeros([convLen - nTap, nRX, nTX, nBatch], 'like', inSig(1))), convLen, 1);
    % change dimension from [Nsample Nrx Ntx Nbatch] to [Nrx Ntx Nsample Nbatch]
    hFPage = permute(hF, [2 3 1 4]);

    % calculate convolution
    yFPage = pagemtimes(hFPage, inSigFPage);
    % change dimension from [Nrx 1 Nsample Nbatch] to [Nsample Nrx Nbatch]
    yF = permute(reshape(yFPage, [nRX convLen nBatch]), [2 1 3]);
    % calculate ifft of YF at first dimension
    y = ifft(yF, convLen, 1);

    fadedSig = y(1:nSample, :, :);
    H = fft(cat(1, h, zeros([fftSize - nTap, nRX, nTX, nBatch], 'like', inSig(1))), fftSize, 1);

end
