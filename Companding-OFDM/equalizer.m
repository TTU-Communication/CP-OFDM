function [outSig] = equalizer(inSig, H, varargin)
    
    narginchk(2, 4);

    validateattributes(inSig, {'double'}, {'2d', 'nonnan', 'finite', 'nonempty'}, ...
                        mfilename, 'Signal', 1);
    validateattributes(H, {'double'}, {'2d', 'nonnan', 'finite', 'nonempty'}, ...
                        mfilename, 'Channel', 2);

    if nargin == 2
        eqMode = 1;   % Zero-Forcing
    elseif nargin == 3
        eqMode = 2;   % MMSE
        noisePower = varargin{1};
        validateattributes(noisePower, {'double'}, ...
            {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Noise_power');
        sigPower = 1;
    elseif nargin == 4
        eqMode = 2;   % MMSE
        noisePower = varargin{1};
        validateattributes(noisePower, {'double'}, ...
            {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Noise_power');
        sigPower = varargin{2};
        validateattributes(sigPower, {'double'}, ...
            {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Signal_power');
    end

    switch (eqMode)
        case 1  % Zero-Forcing
            eq = 1 ./ H;
        case 2  % MMSE
            eq = conj(H) .* (H .* conj(H) + (noisePower ./ sigPower)) .^ -1;
    end

    outSig = eq .* inSig;
end
