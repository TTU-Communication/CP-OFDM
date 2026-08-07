function noise = awgnx(noiseSize, noiseVar, refSig)

    noise = sqrt(noiseVar) .* randn(noiseSize, 'like', refSig);

end
