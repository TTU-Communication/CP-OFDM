function [outSig] = subcarrierDemapping(sig, pilotSCIdx, varargin)
% SUBCARRIERDEMAPPING Extracts data subcarriers from OFDM symbols.
%
%   OUTSIG = SUBCARRIERDEMAPPING(SIG, PILOTSCIDX) removes pilot subcarriers
%   from the OFDM signal SIG, returning only data subcarriers.
%
%   - SIG: The OFDM symbols, specified as a vector or an N-by-M matrix.
%          If SIG is a matrix, each column represents an independent OFDM
%          symbol.
%
%   - PILOTSCIDX: A vector of indices indicating the positions of pilot
%                 subcarriers.
%
%   OUTSIG = SUBCARRIERDEMAPPING(SIG, PILOTSCIDX, NULLSCIDX) also removes
%   subcarriers at the indices specified in NULLSCIDX, corresponding to
%   null subcarriers.
%
%   Output:
%
%   - OUTSIG: A signal containing only data subcarriers, with pilot and
%             null subcarriers removed.
    
    narginchk(2, 3);

    validateattributes(sig, {'numeric'}, {'2d', 'finite'}, mfilename, 'Sig', 1);

    if isrow(sig)
        newSig = sig.';
    else
        newSig = sig;
    end

    fftSize = size(newSig, 1);
    validIdx = {'vector', 'positive', '<=', fftSize};
    validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx', 2);

    [nullSCIdx] = validInputArgs(fftSize, varargin{:});

    if any(ismember(nullSCIdx, pilotSCIdx))
        error("Some of the indices of null subcarriers and pilot subcarriers are the same.");
    end

    sigIdx = setdiff(1:fftSize, [nullSCIdx pilotSCIdx]);
    outSig = newSig(sigIdx, :);

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
