function varargout = scDemap(x, nfft, varargin)
% SCDEMAP Extracts data and pilot subcarriers from OFDM symbols.
%
%   Y = SCDEMAP(X, NFFT) extracts all subcarriers from the input OFDM
%   signal X, assuming no null or pilot subcarriers are present.
%
%   - X: The OFDM symbols, specified as a vector or an NFFT-by-M matrix.
%        If X is a matrix, each column represents an independent OFDM
%        symbol.
%
%   - NFFT: The total number of subcarriers in the OFDM system.
%
%   Y = SCDEMAP(X, NFFT, NULLIDX) removes subcarriers at the indices
%   specified in NULLIDX, corresponding to null subcarriers.
%
%   Y = SCDEMAP(X, NFFT, NULLIDX, PILOTIDX) further removes pilot
%   subcarriers at the indices specified in PILOTIDX.
%
%   [Y, PILOTS] = SCDEMAP(X, NFFT, NULLIDX, PILOTIDX) additionally returns
%   the values extracted from the pilot subcarrier positions.
%
%   Output:
%
%   - Y: The resulting signal containing only data subcarriers, with null
%        and pilot subcarriers removed.
%
%   - PILOTS: A matrix or vector containing the pilot subcarrier values
%             extracted from X at the positions specified by PILOTIDX.

    narginchk(3, 4);

    [prmStr, dataIdx] = setup(x, nfft, varargin{:});

    varargout{1} = x(dataIdx, :, :);

    if ~isempty(prmStr.PilotIndices)
        varargout{2} = x(prmStr.PilotIndices, :, :);
    else
        nargoutchk(0, 1);
    end
    
end

function [prmStr, pDataIdx] = setup(x, nfft, varargin)

    validateattributes(x, {'numeric'}, ...
        {'3d', 'nonempty', 'finite'}, mfilename, 'X', 1);

    [~, numSym, numRX] = size(x);

    validateattributes(nfft, {'numeric'}, ...
        {'real', 'integer', 'scalar', 'positive', 'nonempty', 'finite'}, ...
        mfilename, 'NFFT', 2);

    if isempty(varargin)
        NullIndices = [];
        PilotIndices = [];
        hasPilots = false;

    elseif length(varargin) == 1
        NullIndices = varargin{1};
        PilotIndices = [];
        hasPilots = false;

    elseif length(varargin) == 2
        NullIndices = varargin{1};
        PilotIndices = varargin{2};
        hasPilots = true;

    end

    prmStr = struct(...
        "FFTLength", nfft, ...
        "NumSymbols", numSym, ...
        "numRXs", numRX, ...
        "NullIndices", NullIndices, ...
        "PilotIndices", PilotIndices, ...
        "hasPilots", hasPilots);

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
