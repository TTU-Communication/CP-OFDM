function [channel] = channelEstimator(sig, pilotSCIdx, pilotValue, varargin)
% CHANNELESTIMATOR Estimate the channel response from pilot subcarriers
%
%   CHANNEL = CHANNELESTIMATOR(SIG, PILOTSCIDX, PILOTVALUE) estimates the
%   channel response using the received signal SIG, the indices of pilot
%   subcarriers PILOTSCIDX, and their corresponding transmitted values
%   PILOTVALUE.
%
%   - SIG: The received signal, either as a vector or a 2D matrix.
%          If SIG is a matrix, each column is treated as an independent
%          signal.
%
%   - PILOTSCIDX: A vector containing the indices of the pilot subcarriers.
%
%   - PILOTVALUE: A vector containing the known transmitted values at the
%                 pilot subcarriers.
%
%   CHANNEL = CHANNELESTIMATOR(SIG, PILOTSCIDX, PILOTVALUE, NULLSCIDX)
%   allows you to specify indices of null subcarriers using NULLSCIDX.
%   NULLSCIDX is a vector and can be empty if there are no null
%   subcarriers.
%
%   Output:
%
%   - CHANNEL: The estimated channel response corresponding to the
%              subcarriers in SIG.

    narginchk(4, 5);

    validateattributes(sig, {'numeric'}, {'2d', 'finite'}, mfilename, 'Sig', 1);

    if isrow(sig)
        newSig = sig.';
    else
        newSig = sig;
    end

    [fftSize, sigCol] = size(newSig);
    validIdx = {'vector', 'positive', '<=', fftSize};
    validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx', 2);
    validateattributes(pilotValue, {'numeric'}, {'vector', 'finite'}, mfilename, 'PilotValue', 3);

    [nullSCIdx] = validInputArgs(fftSize, varargin{:});

    if any(ismember(nullSCIdx, pilotSCIdx))
        error("Some of the indices of null subcarriers and pilot subcarriers are the same.");
    end

    % Prepare estimate channel
    pilotValueLen = length(pilotValue);
    dataSCIdx = setdiff(1:fftSize, [nullSCIdx pilotSCIdx]);
    channel = nan(size(newSig));

    for idx = 1:sigCol
        tempPilotValueIdx = (1:length(pilotSCIdx)) + idx - 1;
        tempPilotValue = pilotValue(mod(tempPilotValueIdx - 1, pilotValueLen) + 1);
        tempPilotValue = reshape(tempPilotValue, length(pilotSCIdx), 1);
        pilotChannel = newSig(pilotSCIdx, idx) ./ tempPilotValue; % Get channel information from pilot

        % Channel estimation
        F = griddedInterpolant(pilotSCIdx, pilotChannel);
        channel(pilotSCIdx, idx) = pilotChannel;
        channel(dataSCIdx, idx) = F(dataSCIdx);
    end

    channel(nullSCIdx, :) = 1;

    if isrow(sig)
        channel = channel.';
    end
end

function [nullSCIdx] = validInputArgs(fftSize, varargin)
    
    validIdx = {'vector', 'positive', '<=', fftSize};

    nInArgs = nargin;
    if nInArgs == 1
        nullSCIdx = [];

    elseif nInArgs == 2
        nullSCIdx = varargin{1};

        if ~isempty(nullSCIdx)
            validateattributes(nullSCIdx, {'numeric'}, validIdx, mfilename, 'NullSCIdx');
        end

    end

end
