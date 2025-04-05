function [outSig] = subcarrierDemapping(sig, varargin)
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
    %     - nullSCIdx   : Indices of the null subcarriers (e.g., DC or
    %                     guard bands). Can be empty if no null subcarriers
    %                     are used.
    %     - pilotSCIdx  : Indices of the pilot subcarriers.
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

    fftSize = size(sig, 1);
    validateattributes(sig, {'numeric'}, {'2d', 'finite'}, mfilename, 'Sig', 1);

    [nullSCIdx, pilotSCIdx] = validInputArgs(fftSize, varargin{:});

    if any(ismember(nullSCIdx, pilotSCIdx))
        error("Some of the indices of null subcarriers and pilot subcarriers are the same.");
    end

    sigIdx = setdiff(1:fftSize, [nullSCIdx pilotSCIdx]);
    outSig = sig(sigIdx, :);
end

function [nullSCIdx, pilotSCIdx] = validInputArgs(fftSize, varargin)
    
    validIdx = {'vector', 'positive', '<=', fftSize};

    nInArgs = nargin;
    if nInArgs == 2
        nullSCIdx = [];
        pilotSCIdx = varargin{1};
        
        validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx');

    elseif nInArgs == 3
        nullSCIdx = varargin{1};
        pilotSCIdx = varargin{2};

        if ~isempty(nullSCIdx)
            validateattributes(nullSCIdx, {'numeric'}, validIdx, mfilename, 'NullSCIdx');
        end
        validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx');

    end

end
