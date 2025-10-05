function [outSig] = equalizer(inSig, H, noisePower, sigPower)
    
    arguments
        inSig (:,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty}
        H (:,:,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty}
        noisePower (1,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty} = 0
        sigPower (1,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty} = 1
    end

    [sigSample, sigBatch, nRX] = size(inSig);
    nTX = size(H, 4);

    % change dimension from [Nsp Ns Nrx] to [Nrx 1 Nsp Ns]
    inSigPage = reshape(permute(inSig, [3 1 2]), [nRX 1 sigSample  sigBatch]);
    % change dimension from [Nsp Ns Nrx Ntx] to [Nrx Ntx Nsp Ns]
    HPage = permute(H, [3 4 1 2]);

    if noisePower(1) ~= 0
        % change dimension from [1 Ns Nrx] to [Nrx 1 Nsp Ns]
        noiseVar = reshape(permute(repmat(noisePower, [sigSample 1 1]), [3 1 2]), [nRX 1 sigSample sigBatch]);
        if ndims(noisePower) == 3
            % diagonal matrices based on every column vector
            noiseVar = eye(nRX) .* noiseVar;
        end
        % calculate the inverse matrix
        noiseVarInv = pageinv(noiseVar);
    else
        % set to identity matrix for each signals
        noiseVar = 0;
        noiseVarInv = repmat(eye(nRX), [1 1 sigSample sigBatch]);
    end

    if sigPower(1) ~= 1
        % change dimension from [1 Ns Ntx] to [Ntx 1 Nsp Ns]
        sigVar = reshape(permute(repmat(sigPower, [sigSample 1 1]), [3 1 2]), [nTX 1 sigSample sigBatch]);
        if ndims(sigPower) == 3
            % diagonal matrices based on every column vector
            sigVar = eye(nTX) .* sigVar;
        end
        % calculate the inverse matrix
        sigVarInv = pageinv(sigVar);
    elseif noisePower(1) ~= 0
        % set to identity matrix for each signals if noisePower is provided
        sigVarInv = repmat(eye(nTX), [1 1 sigSample sigBatch]);
    else
        sigVar = 1;
        sigVarInv = 0;
    end

    if size(HPage, 1) >= size(HPage, 2)
        % left inverse (generalized inverse)
        % using $(H^H R_w^-1 H + R_x)^-1 H^H R_w^-1$
        HPageCTNoiseVarInv = pagemtimes(HPage, 'ctranspose', noiseVarInv, 'none');
        leftInvMtx = pagemtimes(HPageCTNoiseVarInv, HPage) + sigVarInv;
        preEQ = pagemldivide(leftInvMtx, pagectranspose(HPage));
        eq = pagemtimes(preEQ, noiseVarInv);
        % eq = pagemldivide(pagemtimes(HPage, 'ctranspose', HPage, 'none') ...
        %     + (noisePower ./ sigPower), pagectranspose(HPage));
    else
        % right inverse
        % using $R_x H^H (H R_x H^H + R_w)^-1$
        HPageSigVar = pagemtimes(HPage, sigVar);
        rightInvMtx = pagemtimes(HPageSigVar, 'none', HPage, 'ctranspose') + noiseVar;
        preEQ = pagemtimes(sigVar, 'none', HPage, 'ctranspose');
        eq = pagemrdivide(preEQ, rightInvMtx);
        % eq = pagemrdivide(pagectranspose(HPage), pagemtimes(HPage, 'none', HPage, 'ctranspose') ...
        %     + (noisePower ./ sigPower));
    end

    outSigPage = pagemtimes(eq, inSigPage);
    % change dimension from [Ntx 1 Nsp Ns] to [Nsp Ns Ntx]
    outSig = permute(reshape(outSigPage, [nTX sigSample sigBatch]), [2 3 1]);
end
