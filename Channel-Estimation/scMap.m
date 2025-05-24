function y = scMap(x, nfft, varargin)
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
    
    narginchk(3, 5);

    [prmStr, dataIdx] = setup(x, nfft, varargin{:});

    FFTLen = prmStr.FFTLength;
    numSym = prmStr.NumSymbols;

    if isempty(prmStr.Pilots)
        typeIn = cast(1i, "like", x);
    else
        typeIn = cast(1i, "like", prmStr.Pilots(1) + x(1));
    end

    y = zeros(FFTLen, numSym, 'like', typeIn);
    y(dataIdx, :) = x;
    if ~isempty(prmStr.PilotIndices) && ~isempty(prmStr.Pilots)
        y(prmStr.PilotIndices, :) = prmStr.Pilots;
    end

    y = ifftshift(y, 1);

end

function [prmStr, pDataIdx] = setup(x, nfft, varargin)

    validateattributes(x, {'numeric'}, ...
        {'2d', 'nonempty', 'finite'}, mfilename, 'X', 1);

    [numST, numSym] = size(x);

    validateattributes(nfft, {'numeric'}, ...
        {'real', 'integer', 'scalar', 'positive', 'nonempty', 'finite'}, ...
        mfilename, 'NFFT', 2);

    if isempty(varargin)
        NullIndices = [];
        PilotIndices = [];
        Pilots = [];

    elseif length(varargin) == 1
        NullIndices = varargin{1};
        PilotIndices = [];
        Pilots = [];

    elseif length(varargin) == 2
        error("Invalid input: pilot values not provided.");

    elseif length(varargin) == 3
        NullIndices = varargin{1};
        PilotIndices = varargin{2};
        Pilots = varargin{3};

    end

    prmStr = struct(...
        "FFTLength", nfft, ...
        "NumSymbols", numSym, ...
        "NullIndices", NullIndices, ...
        "PilotIndices", PilotIndices, ...
        "Pilots", Pilots);

    if ~isempty(prmStr.PilotIndices) && isempty(prmStr.Pilots)
        error("Invalid input: provided pilot values are empty.");
    end

    if ~isempty(prmStr.PilotIndices) && ~isempty(prmStr.Pilots)
        prmStr.hasPilots = true;
    else
        prmStr.hasPilots = false;
    end

    if ~isempty(prmStr.NullIndices)
        checkNulls(prmStr, numST);

        dataIdx = double(setdiff((1:nfft)', prmStr.NullIndices));
    else
        if prmStr.hasPilots
            numPilots = length(prmStr.PilotIndices);
            assert(nfft == numST + numPilots, ...
                "Invalid input argument lengths for signal, and pilot indices.");
        else
            assert(nfft == numST, ...
                "Invalid input argument lengths for signal.");
        end

        dataIdx = double((1:nfft)');
    end
    
    if ~isempty(prmStr.PilotIndices)
        checkPilots(prmStr, numSym);

        pDataIdx = setdiff(dataIdx, prmStr.PilotIndices);
    else
        pDataIdx = dataIdx;
    end

end

function checkNulls(prmStr, numST)
    validateattributes(prmStr.NullIndices, {'numeric'}, ...
        {'column', 'real', 'positive', 'integer', 'nonempty', 'finite'}, ...
        mfilename, 'NULLIDX');

    numNulls = length(prmStr.NullIndices);

    assert(length(unique(prmStr.NullIndices)) == numNulls, ...
        "Null indices are not unique.");

    assert(all(prmStr.NullIndices <= prmStr.FFTLength), ...
        "Null indices are larger than FFT length.");

    if prmStr.hasPilots
        numPilots = length(prmStr.PilotIndices);
    else
        numPilots = 0;
    end

    assert(prmStr.FFTLength == numST + numNulls + numPilots, ...
        "Invalid input argument lengths for signal, null indices, and pilot indices.");

end

function checkPilots(prmStr, numSym)
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

    [np, pSym] = size(prmStr.Pilots);

    assert(np == numPilots && pSym == numSym, ...
        'Pilots are not the same size as the pilot indices and symbol counts.');
end
