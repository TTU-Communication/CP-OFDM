function [outSig] = subcarrierMapping(sig, fftSize, pilotSCIdx, pilotValue, varargin)
% SUBCARRIERMAPPING Maps subcarriers for null, pilot, and data signals.
%
%   OUTSIG = SUBCARRIERMAPPING(SIG, FFTSIZE, PILOTSCIDX, PILOTVALUE) maps
%   the data signal SIG and pilot values PILOTVALUE to the appropriate
%   subcarrier indices in an OFDM symbol of size FFTSIZE.
%
%   - SIG: The data signal, provided as a vector or an N-by-M matrix.
%          If SIG is a matrix, each column represents an independent
%          modulated OFDM symbol.
%
%   - FFTSIZE: The total number of subcarriers in the OFDM system.
%
%   - PILOTSCIDX: A vector specifying the indices of the pilot subcarriers.
%
%   - PILOTVALUE: A vector of known pilot symbols to be inserted at the
%                 specified pilot subcarrier indices. These values are
%                 applied cyclically across OFDM symbols to support phase
%                 tracking and frequency offset correction, in accordance
%                 with IEEE 802.11ac/ax standards.
%
%   OUTSIG = SUBCARRIERMAPPING(SIG, FFTSIZE, PILOTSCIDX, PILOTVALUE,
%   NULLSCIDX) also allows you to specify null subcarrier positions using
%   the vector NULLSCIDX. These subcarriers are excluded from data and
%   pilot mapping.
%
%   Output:
%
%   - OUTSIG: The resulting OFDM symbol(s) with pilot, data, and null
%             subcarriers mapped appropriately.
    
    narginchk(4, 5);

    validateattributes(sig, {'numeric'}, {'2d', 'finite'}, mfilename, 'Sig', 1);

    if isrow(sig)
        newSig = sig.';
    else
        newSig = sig;
    end

    validateattributes(fftSize, {'numeric'}, {'scalar', 'positive'}, mfilename, 'FFTSize', 2);
    validIdx = {'vector', 'positive', '<=', fftSize};
    validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx', 3);
    validateattributes(pilotValue, {'numeric'}, {'vector', 'finite'}, mfilename, 'PilotValue', 4);

    [nullSCIdx] = validInputArgs(fftSize, varargin{:});

    if fftSize ~= size(newSig, 1) + length(nullSCIdx) + length(pilotSCIdx)
        error("The input signal length is not the same as the FFT size.");
    end
    if any(ismember(nullSCIdx, pilotSCIdx))
        error("Some of the indices of null subcarriers and pilot subcarriers are the same.");
    end

    sigCol = size(newSig, 2);
    sigIdx = setdiff(1:fftSize, [nullSCIdx pilotSCIdx]);
    tempPilotValueIdx = (1:length(pilotSCIdx)).' + (1:sigCol) - 1;
    tempPilotValueIdx = mod(tempPilotValueIdx - 1, length(pilotValue)) + 1;

    outSig = zeros(fftSize, sigCol);
    outSig(sigIdx, :) = newSig;
    outSig(pilotSCIdx, :) = reshape(pilotValue(tempPilotValueIdx), length(pilotSCIdx), sigCol);

    if isrow(sig)
        outSig = outSig.';
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
