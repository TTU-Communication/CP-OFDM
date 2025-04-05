function [outSig] = subcarrierMapping(sig, fftSize, varargin)
    % SUBCARRIERMAPPING Maps subcarriers for null, pilot, and data signals.
    %
    %   This function inserts OFDM-modulated signals onto appropriate
    %   subcarriers, assigning null, pilot, and data subcarriers based on
    %   specified indices.
    %
    %   Inputs:
    %     - sig         : An N-by-M matrix, where each column is an
    %                     independent modulated signals for OFDM symbols.
    %     - fftSize     : FFT size, i.e., the total number of subcarriers.
    %     - nullSCIdx   : Indices of null subcarriers (e.g., DC subcarrier,
    %                     guard bands). Can be empty if no null subcarriers
    %                     are used.
    %     - pilotSCIdx  : Indices of pilot subcarriers.
    %     - pilotValue  : A vector of pilot values (±1) defined according 
    %                     to the IEEE 802.11ac/ax standard. The values are
    %                     applied cyclically across OFDM symbols to enable
    %                     phase tracking and frequency offset correction.
    %
    %   Output:
    %     - outSig      : A fftSize-by-M matrix where each column
    %                     represents one OFDM symbol with subcarriers
    %                     mapped appropriately.
    %
    %   Note:
    %     This mapping follows the IEEE 802.11 standard where pilot tones
    %     are fixed at ±1 and are used for phase tracking and frequency
    %     offset correction.
    
    narginchk(4, 5);

    validateattributes(sig, {'numeric'}, {'2d', 'finite'}, mfilename, 'Sig', 1);
    validateattributes(fftSize, {'numeric'}, {'scalar', 'positive'}, mfilename, 'FFTSize', 2);

    [nullSCIdx, pilotSCIdx, pilotValue] = validInputArgs(fftSize, varargin{:});

    if fftSize ~= size(sig, 1) + length(nullSCIdx) + length(pilotSCIdx)
        error("The input signal length is not the same as the FFT size.");
    end
    if any(ismember(nullSCIdx, pilotSCIdx))
        error("Some of the indices of null subcarriers and pilot subcarriers are the same.");
    end

    sigCol = size(sig, 2);
    sigIdx = setdiff(1:fftSize, [nullSCIdx pilotSCIdx]);
    tempPilotValueIdx = (1:length(pilotSCIdx)).' + (1:sigCol) - 1;
    tempPilotValueIdx = mod(tempPilotValueIdx - 1, length(pilotValue)) + 1;

    outSig = zeros(fftSize, sigCol);
    outSig(sigIdx, :) = sig;
    outSig(pilotSCIdx, :) = reshape(pilotValue(tempPilotValueIdx), length(pilotSCIdx), sigCol);

end

function [nullSCIdx, pilotSCIdx, pilotValue] = validInputArgs(fftSize, varargin)
    
    validIdx = {'vector', 'positive', '<=', fftSize};

    nInArgs = nargin;
    if nInArgs == 3
        nullSCIdx = [];
        pilotSCIdx = varargin{1};
        pilotValue = varargin{2};

        validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx');
        validateattributes(pilotValue, {'numeric'}, {'2d', 'finite'}, mfilename, 'PilotSCValue');

    elseif nInArgs == 4
        nullSCIdx = varargin{1};
        pilotSCIdx = varargin{2};
        pilotValue = varargin{3};

        if ~isempty(nullSCIdx)
            validateattributes(nullSCIdx, {'numeric'}, validIdx, mfilename, 'NullSCIdx');
        end
        validateattributes(pilotSCIdx, {'numeric'}, validIdx, mfilename, 'PilotSCIdx');
        validateattributes(pilotValue, {'numeric'}, {'2d', 'finite'}, mfilename, 'PilotSCValue');

    end

end
