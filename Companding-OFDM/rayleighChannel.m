function [fadedSig, H] = rayleighChannel(inSig, nTap, tapPower, fftSize)

    [sigRow, sigCol] = size(inSig);
    h = sqrt(tapPower) * randn(nTap, sigCol, 'like', inSig(1));
    convLen = sigRow + nTap - 1;

    y = ifft(fft(inSig, convLen, 1) .* fft(h, convLen, 1), convLen, 1);

    fadedSig = y(1:sigRow, :);
    H = fft(h, fftSize, 1);

end
