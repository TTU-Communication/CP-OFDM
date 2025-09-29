function y = scMap(x, nfft, varargin)
% SCMAP Maps subcarriers for data, null, and pilot signals in OFDM.
%
%   Y = SCMAP(X, NFFT) maps the data signal X to the available subcarriers
%   in an OFDM symbol of size NFFT.
%
%   - X: The data signal, specified as a vector or an N-by-M matrix.
%        If X is a matrix, each column represents an independent modulated
%        OFDM symbol.
%
%   - NFFT: The total number of subcarriers in the OFDM system.
%
%   Y = SCMAP(X, NFFT, NULLIDX) excludes subcarriers at positions specified
%   in NULLIDX from mapping. NULLIDX is a vector of null subcarrier
%   indices.
%
%   Y = SCMAP(X, NFFT, NULLIDX, PILOTIDX, PILOTS) additionally inserts
%   pilot values PILOTS at the subcarrier positions specified by PILOTIDX.
%
%   Output:
%
%   - Y: The resulting OFDM symbols with data, null, and pilot subcarriers
%        mapped appropriately.
    
    narginchk(3, 5);

    [prmStr, dataIdx] = setup(x, nfft, varargin{:});

    FFTLen = prmStr.FFTLength;
    numSym = prmStr.NumSymbols;
    numTX = prmStr.NumTXs;

    if isempty(prmStr.Pilots)
        typeIn = cast(1i, "like", x);
    else
        typeIn = cast(1i, "like", prmStr.Pilots(1) + x(1));
    end

    y = zeros([FFTLen numSym numTX], 'like', typeIn);
    y(dataIdx, :, :) = x;
    if ~isempty(prmStr.PilotIndices) && ~isempty(prmStr.Pilots)
        y(prmStr.PilotIndices, :) = prmStr.Pilots;
    end

    y = ifftshift(y, 1);

end

function [prmStr, pDataIdx] = setup(x, nfft, varargin)

    validateattributes(x, {'numeric'}, ...
        {'3d', 'nonempty', 'finite'}, mfilename, 'X', 1);

    [numST, numSym, numTX] = size(x);

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
        "NumTXs", numTX, ...
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
