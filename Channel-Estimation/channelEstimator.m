function [hEst] = channelEstimator(h, nfft, nullIdx, pilotIdx)
% CHANNELESTIMATOR Estimate the channel response from pilot subcarriers
%
%   HEST = CHANNELESTIMATOR(H, NFFT, NULLIDX, PILOTIDX) interpolates the
%   channel response using the channel response H estimated from pilot
%   subcarriers, the size of FFT NFFT, the indices of null subcarriers
%   NULLIDX, and the indices of pilot subcarriers PILOTIDX.
%
%   - H: The estimated channel response at the pilot subcarriers positions
%        specified in PILOTIDX.
%
%   - NFFT: The total number of subcarriers in the OFDM system.
%
%   - NULLIDX: A column vector containg the indices of the null
%              subcarriers, which are excluded from channel estimation.
%
%   - PILOTIDX: A column vector containing the indices of the pilot
%                 subcarriers.
%
%   Output:
%
%   - HEST: The estimated channel response corresponding to the FFT indices
%           excludes NULLIDX and PILOTIDX.

    narginchk(4, 4);

    [prmStr, dataIdx] = setup(h, nfft, nullIdx, pilotIdx);

    hEst = zeros([length(dataIdx) 1 1 prmStr.NumBatch], 'like', h(1));

    for idx = 1:prmStr.NumBatch
        % Channel estimation
        F = griddedInterpolant(prmStr.PilotIndices, h(:, :, :, idx));
        hEst(:, :, :, idx) = F(dataIdx);
    end

end

function [prmStr, pDataIdx] = setup(h, nfft, NullIndices, PilotIndices)

    validateattributes(h, {'numeric'}, ...
        {'nonempty', 'finite'}, mfilename, 'H', 1);

    [~, ~, ~, numBatch]  = size(h);

    validateattributes(nfft, {'numeric'}, ...
        {'real', 'integer', 'scalar', 'positive', 'nonempty', 'finite'}, ...
        mfilename, 'NFFT', 2);

    prmStr = struct(...
        "FFTLength", nfft, ...
        "NumBatch", numBatch, ...
        "NullIndices", NullIndices, ...
        "PilotIndices", PilotIndices);

    if ~isempty(prmStr.NullIndices)
        checkNulls(prmStr);

        dataIdx = double(setdiff((1:nfft)', prmStr.NullIndices));
    else
        dataIdx = double((1:nfft)');
    end
    
    if ~isempty(prmStr.PilotIndices)
        checkPilots(prmStr);

        pDataIdx = setdiff(dataIdx, prmStr.PilotIndices);
    else
        pDataIdx = dataIdx;
    end

end

function checkNulls(prmStr)
    validateattributes(prmStr.NullIndices, {'numeric'}, ...
        {'column', 'real', 'positive', 'integer', 'nonempty', 'finite'}, ...
        mfilename, 'NULLIDX');

    numNulls = length(prmStr.NullIndices);

    assert(length(unique(prmStr.NullIndices)) == numNulls, ...
        "Null indices are not unique.");

    assert(all(prmStr.NullIndices <= prmStr.FFTLength), ...
        "Null indices are larger than FFT length.");

end

function checkPilots(prmStr)
    validateattributes(prmStr.PilotIndices, {'numeric'}, ...
        {'column', 'real', 'positive', 'integer', 'nonempty', 'finite'}, ...
        mfilename, 'PILOTIDX');

    numPilots = length(prmStr.PilotIndices);

    assert(length(unique(prmStr.PilotIndices)) == numPilots, ...
        "Pilot indices are not unique.");

    assert(all(prmStr.PilotIndices <= prmStr.FFTLength), ...
        "Pilot indices are larger than FFT length.");

    numNulls = length(prmStr.NullIndices);
    assert(length(unique([prmStr.PilotIndices; ...
        prmStr.NullIndices])) == (numPilots + numNulls), ...
        'Null and Pilot indices are not unique.');

end
