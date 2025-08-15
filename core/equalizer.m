function [outSig] = equalizer(inSig, H, noisePower, sigPower)
    
    arguments
        inSig (:,:,:) {mustBeFinite, mustBeNonempty}
        H (:,:,:,:) {mustBeFinite, mustBeNonempty}
        noisePower (1,:,:) {mustBeFinite, mustBeNonempty} = 0
        sigPower (1,:,:) {mustBeFinite, mustBeNonempty} = 1
    end

    [sigSample, sigBatch, nRX] = size(inSig);
    nTX = size(H, 4);

    % change dimension from [Nsp Ns Nrx] to [Nrx 1 Nsp Ns]
    inSigPage = reshape(permute(inSig, [3 1 2]), [nRX 1 sigSample  sigBatch]);
    % change dimension from [Nsp Ns Nrx Ntx] to [Nrx Ntx Nsp Ns]
    HPage = permute(H, [3 4 1 2]);
    if noisePower(1) ~= 0
        noisePower = permute(repmat(noisePower, [sigSample 1 1]), [3 1 2]);
        if ndims(noisePower) == 3
            noisePower = eye(nRX) .* reshape(noisePower, [nRX 1 sigSample sigBatch]);
        end
    end
    if sigPower(1) ~= 1
        sigPower = permute(repmat(sigPower, [sigSample 1 1]), [3 1 2]);
        if ndims(sigPower) == 3
            sigPower = eye(nRX) .* reshape(sigPower,[nRX 1 sigSample sigBatch]);
        end
    end

    eq = pagemrdivide(pagectranspose(HPage), pagemtimes(HPage, 'none', HPage, 'ctranspose') ...
            + (noisePower ./ sigPower));

    outSigPage = pagemtimes(eq, inSigPage);
    outSig = permute(reshape(outSigPage, [nTX sigSample sigBatch]), [2 3 1]);
end
