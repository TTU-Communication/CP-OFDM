function varargout = scDemap(x, nfft, varargin)
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
    
    narginchk(3, 4);

    [prmStr, dataIdx] = setup(x, nfft, varargin{:});

    preShiftx = fftshift(x, 1);

    varargout{1} = preShiftx(dataIdx, :);

    if ~isempty(prmStr.PilotIndices)
        varargout{2} = preShiftx(prmStr.PilotIndices, :);
    else
        nargoutchk(0, 1);
    end
    
end

function [prmStr, pDataIdx] = setup(x, nfft, varargin)

    validateattributes(x, {'numeric'}, ...
        {'2d', 'nonempty', 'finite'}, mfilename, 'X', 1);

    [~, numSym] = size(x);

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
