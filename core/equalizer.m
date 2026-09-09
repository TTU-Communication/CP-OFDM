function [outSig] = equalizer(inSig, H, noisePower, sigPower)
    
    arguments
        % Input signal and it's dimension should be [Nsample Nsymbol Nrx Nbatch]
        inSig (:,:,:, :) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty}
        % Channel matrix: [Nsample Nrx Ntx Nbatch]
        % Scalar H = 1 indicates no channel distortion and bypasses equalization.
        H (:,:,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty, mustBeValidChannel(H, inSig)}
        % noisePower and sigPower are optional.
        % If noisePower is omitted:
        %   nRX >= nTX: ZF spatial equalizer
        %   nRX <  nTX: minimum-norm Moore-Penrose estimator
        % noisePower's dimension should be [1 Nrx Nbatch] or [1 1 Nbatch] or [1 1 1]
        noisePower (1,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeReal, mustBePositive} = []
        % sigPower's dimension should be [1 Ntx Nbatch] or [1 1 Nbatch] or [1 1 1]
        sigPower (1,:,:) {mustBeArrayOrGPU, mustBeFinite, mustBeReal, mustBePositive} = []
    end

    if isempty(noisePower) && ~isempty(sigPower)
        error("Equalizer:SignalPowerWithoutNoisePower", ...
            "sigPower can only be used when noisePower is provided.");
    end

    if isscalar(H) && H == 1
        outSig = inSig;
        return
    end

    [~, ~, nRX, nBatch] = size(inSig);
    nTX = size(H, 3);

    % change dimension from [Nsample Nsymbol Nrx Nbatch] to [Nrx Nsymbol Nsample Nbatch]
    inSigPage = permute(inSig, [3 2 1 4]);
    % change dimension from [Nsample Nrx Ntx Nbatch] to [Nrx Ntx Nsample Nbatch]
    HPage = permute(H, [2 3 1 4]);

    if ~isempty(noisePower)
        % change dimension from [1 Nrx Nbatch] to [Nrx Nrx 1 Nbatch]
        [noiseVar, noiseVarInv] = makeDiagonalCovariance(...
                noisePower, nRX, nBatch, HPage(1));
    else
        % set to identity matrix for each signals
        noiseVar = 0;
        noiseVarInv = eye(nRX, 'like', HPage(1));
    end

    if ~isempty(sigPower)
        % change dimension from [1 Ntx Nbatch] to [Ntx Ntx 1 Nbatch]
        [sigVar, sigVarInv] = makeDiagonalCovariance(...
                sigPower, nTX, nBatch, HPage(1));
    elseif ~isempty(noisePower)
        % set to identity matrix for each signals if noisePower is provided
        sigVar = eye(nTX, 'like', HPage(1));
        sigVarInv = sigVar;
    else
        sigVar = eye(nTX, "like", HPage(1));
        sigVarInv = 0;
    end

    if nRX >= nTX
        % Information-form LMMSE
        % using $(H^H R_w^-1 H + R_x^-1)^-1 H^H R_w^-1$
        HPageCTNoiseVarInv = pagemtimes(HPage, 'ctranspose', noiseVarInv, 'none');
        leftInvMtx = pagemtimes(HPageCTNoiseVarInv, HPage) + sigVarInv;
        eq = pagemldivide(leftInvMtx, HPageCTNoiseVarInv);
        % eq = pagemldivide(pagemtimes(HPage, 'ctranspose', HPage, 'none') ...
        %     + (noisePower ./ sigPower), pagectranspose(HPage));
    else
        % Covariance-form LMMSE
        % using $R_x H^H (H R_x H^H + R_w)^-1$
        SigVarHPageCT = pagemtimes(sigVar, 'none', HPage, 'ctranspose');
        rightInvMtx = pagemtimes(HPage, SigVarHPageCT) + noiseVar;
        eq = pagemrdivide(SigVarHPageCT, rightInvMtx);
        % eq = pagemrdivide(pagectranspose(HPage), pagemtimes(HPage, 'none', HPage, 'ctranspose') ...
        %     + (noisePower ./ sigPower));
    end

    outSigPage = pagemtimes(eq, inSigPage);
    % change dimension from [Ntx Nsymbol Nsample Nbatch] to [Nsample Nsymbol Ntx Nbatch]
    outSig = permute(outSigPage, [3 2 1 4]);
end

function [cov, covInv] = makeDiagonalCovariance(powerVal, nAnt, nBatch, refSample)

    nPowerAnt = size(powerVal, 2);
    nPowerBatch = size(powerVal, 3);

    if nPowerAnt ~= 1 && nPowerAnt ~= nAnt
        error("Equalizer:PowerAntennaMismatch", ...
            "Power antenna size must be 1 or match the antenna count.");
    end

    if nPowerBatch ~= 1 && nPowerBatch ~= nBatch
        error("Equalizer:PowerBatchMismatch", ...
            "Power batch size must be 1 or match the signal batch size.");
    end

    % [1, nPowerAnt, nPowerBatch]
    % -> [nPowerAnt, 1, 1, nPowerBatch]
    powerVec = reshape( ...
        permute(powerVal, [2 1 3]), ...
        [nPowerAnt, 1, 1, nPowerBatch]);

    powerVec = cast(powerVec, "like", refSample);

    identityMatrix = eye(nAnt, "like", refSample);

    cov = identityMatrix .* powerVec;
    covInv = identityMatrix .* (1 ./ powerVec);

end

function mustBeValidChannel(H, inSig)

    % Scalar H = 1 denotes a bypass channel:
    % no channel equalization is required.
    if isscalar(H) && H == 1
        return
    end

    [nSample, ~, nRX, nBatch] = size(inSig);

    if size(H, 1) ~= nSample
        error("Equalizer:SampleMismatch", ...
            "inSig and H must have the same sample/subcarrier count.");
    end

    if size(H, 2) ~= nRX
        error("Equalizer:RxMismatch", ...
            "The RX dimensions of inSig and H must agree.");
    end

    nChannelBatch = size(H, 4);

    if nChannelBatch ~= 1 && nChannelBatch ~= nBatch
        error("Equalizer:BatchMismatch", ...
            "H batch size must be 1 or match inSig.");
    end
end
