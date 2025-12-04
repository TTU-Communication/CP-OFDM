function [nullIdx] = getNullIdx(nfft, length, DCLength)
% GETNULLIDX  Return indices of null subcarriers for a given FFT size.
%
% SYNTAX
%   NULLIDX = GETNULLIDX(NFFT)
%   NULLIDX = GETNULLIDX(NFFT, LENGTH)
%   NULLIDX = GETNULLIDX(NFFT, LENGTH, DCLENGTH)
%
% DESCRIPTION
%   NULLIDX = GETNULLIDX(NFFT) returns the indices of null subcarriers
%   (guard bands and DC) according to IEEE 802.11 for supported FFT sizes:
%   32, 64, 128, 256, 512, 1024.
%
%   NULLIDX = GETNULLIDX(NFFT, LENGTH) returns user-defined null subcarrier
%   indices. LENGTH is the total number of guard-band tones (two-sided).
%   The function splits LENGTH approximately in half so that data
%   subcarriers are centered around DC. When LENGTH is odd, the negative-
%   frequency side (in a -pi...pi view) gets one more null tone than the
%   positive side.
%
%   NULLIDX = GETNULLIDX(NFFT, LENGTH, DCLENGTH) also allows a user-defined
%   DC region. DCLENGTH specifies the number of DC tones (default 0) and is
%   counted as part of LENGTH. The DC region is centered around 0
%   frequency. When DCLENGTH is odd, the positive-frequency side (in a
%   -pi...pi view) has one more null tone than the negative side to keep DC
%   centered.
%
% INPUTS
%   NFFT      - FFT size. Supported values for IEEE 802.11 presets:
%               32, 64, 128, 256, 512, 1024.
%   LENGTH    - (Optional) Total number of guard-band null tones (two-sided).
%               If omitted, standard 802.11 allocation is used.
%   DCLENGTH  - (Optional) Width (in tones) of the DC null region; counted
%               inside LENGTH. If omitted, the default DC width is used.
%
% OUTPUTS
%   NULLIDX   - Column vector of FFT-bin indices corresponding to null
%               subcarriers (guard bands + DC), ordered consistently with
%               1...NFFT.
%
% NOTES
% * LENGTH >= DCLENGTH >= 0 must hold for custom allocations.
% * For IEEE presets, only the specified NFFT values are supported.
%
% EXAMPLES
%   % IEEE 802.11 (e.g., 64-point) null subcarrier indices:
%   idx = GETNULLIDX(64);
%
%   % Custom: total guard-band length 12 (two-sided), 0 DC width:
%   idx = GETNULLIDX(128, 12);
%
%   % Custom: guard bands total 14 tones, DC region 1 tone wide:
%   idx = GETNULLIDX(256, 14, 1);
    
    narginchk(1, 3);

    validateattributes(nfft, {'numeric'}, {"scalar", "integer", 'positive'}, mfilename, "NFFT", 1);

    if nargin == 1

        if log2(nfft) ~= floor(log2(nfft))
            error("FFT size must be the power of 2");
        end

        switch(nfft)
            case 32 % IEEE 802.11ah
                negNullIdx = -16:-14;
                posNullIdx = [0 14:15];
            case 64 % IEEE 802.11ac
                negNullIdx = -32:-29;
                posNullIdx = [0 29:31];
            case 128 % IEEE 802.11ac
                negNullIdx = [-64:-59 -1];
                posNullIdx = [0:1 59:63];
            case 256 % IEEE 802.11ax
                negNullIdx = [-128:-123 -1];
                posNullIdx = [0:1 123:127];
            case 512 % IEEE 802.11ax
                negNullIdx = [-256:-245 -2:-1];
                posNullIdx = [0:2 245:255];
            case 1024 % IEEE 802.11ax
                negNullIdx = [-512:-501 -2:-1];
                posNullIdx = [0:2 501:511];
            otherwise
                error("FFT size is not support: %d", nfft);
        end

    else
        validateattributes(length, {'numeric'}, {"scalar", "integer", '<', nfft}, mfilename, "LENGTH", 2);
        if nargin == 3
            validateattributes(DCLength, {'numeric'}, {"scalar", "integer", '<', length}, mfilename, "DCLENGTH", 3);
        else
            DCLength = 0;
        end

        nfftHalf = nfft / 2;
        length = length - DCLength;
        negHalf = ceil(length / 2);
        posHalf = length - negHalf;

        negNullIdx = -nfftHalf:(-nfftHalf + negHalf - 1);
        posNullIdx = (nfftHalf - posHalf):(nfftHalf - 1);

        if DCLength > 0
            DCPosHalf = ceil(DCLength / 2);
            DCNegHalf = DCLength - DCPosHalf;

            negNullIdx = [negNullIdx -DCNegHalf:-1];
            posNullIdx = [0:(DCPosHalf-1) posNullIdx];
        end
    end

    % Turn to matlab index
    nullIdx = [posNullIdx (negNullIdx + nfft)]' + 1;
end

