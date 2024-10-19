function [out_sig] = equalizer(sig, ch, varargin)
    
    narginchk(2, 4);

    validateattributes(sig, {'double'}, ...
        {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Signal', 1);
    validateattributes(ch, {'double'}, ...
        {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Channel', 2);

    if nargin == 2
        mode = 1;   % Zero-Forcing
    elseif nargin == 3
        mode = 2;   % MMSE
        noise_power = varargin{1};
        validateattributes(noise_power, {'double'}, ...
            {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Noise_power');
        signal_power = 1;
    elseif nargin == 4
        mode = 2;   % MMSE
        noise_power = varargin{1};
        validateattributes(noise_power, {'double'}, ...
            {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Noise_power');
        signal_power = varargin{2};
        validateattributes(signal_power, {'double'}, ...
            {'2d', 'nonnan', 'finite', 'nonempty'}, mfilename, 'Signal_power');
    end

    switch (mode)
        case 1  % Zero-Forcing
            eq = 1 ./ ch;
        case 2  % MMSE
            eq = conj(ch) .* (ch .* conj(ch) + (noise_power ./ signal_power)) .^ -1;
    end

    out_sig = eq .* sig;
end
