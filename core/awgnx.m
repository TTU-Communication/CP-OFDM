function [noise, var] = awgnx(noiseSize, snrDb, sigPower, refSig)

    snrLinear = 10 .^ (snrDb / 10);
    if length(noiseSize) > 2
        % average noise power to every RX from the sum of TX signal power
        noisePower = sigPower / (snrLinear * noiseSize(3));
        var = repmat(noisePower, 1, 1, noiseSize(3));
    else
        noisePower = sigPower / snrLinear;
        var = noisePower;
    end

    noise = sqrt(noisePower) .* randn(noiseSize, 'like', refSig);

end
