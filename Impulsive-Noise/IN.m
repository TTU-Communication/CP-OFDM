function [noise, happenIdx] = IN(noiseSize, noiseVar, probability, refSig)

    happenIdx = rand(noiseSize, 'like', real(refSig)) < probability;
    noise = sqrt(noiseVar) .* randn(noiseSize, 'like', refSig);
    noise(~happenIdx) = 0;

end
