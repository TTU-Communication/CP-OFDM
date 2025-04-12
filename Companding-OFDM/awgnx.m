function [noise] = awgnx(noiseSize, snrDb, sigPower, refSig)

    snrLinear = 10 .^ (snrDb / 10);
    noisePower = sigPower / snrLinear;
    noise = sqrt(noisePower) .* randn(noiseSize, 'like', refSig);

end
