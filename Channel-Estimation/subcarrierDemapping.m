function [outSig] = subcarrierDemapping(sig, pilotSCIdx, varargin)
    %SUBCARRIERDEMAPPING Extracts data and pilot subcarriers from received
    %   OFDM symbols.
    %
    %   This function performs the inverse of subcarrier mapping by
    %   removing the null and pilot subcarriers from a received OFDM symbol
    %   matrix, leaving only the data subcarriers.
    %
    %   Inputs:
    %     - sig         : A matrix of size N-by-M, where each column
    %                     represents one received OFDM symbol including
    %                     data, pilot, and null subcarriers.
    %     - pilotSCIdx  : Indices of the pilot subcarriers.
    %     - nullSCIdx   : Indices of the null subcarriers (e.g., DC or
    %                     guard bands). Can be empty if no null subcarriers
    %                     are used.
    %
    %   Outputs:
    %     - outSig      : Matrix containing the data subcarriers, after
    %                     removal of null and pilot subcarriers.
    %
    %   Note:
    %     The remaining subcarriers after removing the nulls and pilots are
    %     considered data. The input indices should match the original
    %     mapping used in the transmission stage.
    
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
