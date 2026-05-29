function [noise, var, happenIdx] = IN(noiseSize, snrDb, probability, sigPower, refSig)

    snrLinear = 10 .^ (snrDb / 10);
    if length(noiseSize) > 2
        % average noise power to every RX from the sum of TX signal power
        noisePower = sigPower / (snrLinear * noiseSize(3));
        var = repmat(noisePower, 1, 1, noiseSize(3));
    else
        noisePower = sigPower / snrLinear;
        var = noisePower;
    end

    happenIdx = rand(noiseSize) < probability;
    noise = sqrt(noisePower) .* randn(noiseSize, 'like', refSig);
    noise(~happenIdx) = 0;

end
