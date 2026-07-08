function [restoredSig, process] = PGIR(sigTarget, sigRef, dataMask, transMask, stopCriteria)
    % GPGA Generalized Papoulis-Gerberg Algorithm.
    %   [NMSE_c, iterations_c] = GPGA(yn, xn, pos_Kd, pos_KT, del_NMSE_t)
    %   processes the distroted signal yn with the original signal xn. The 
    %   unknown positions in the Data Domain pos_Kd and the known positions
    %   in the Transform Domain pos_KT are vectors containing only ones and 
    %   zeros. The parameter del_NMSE_t is used to stop the GPGA process 
    %   when the absolution value of the current NMSE minus the previous 
    %   NMSE is less than del_NMSE_t.
    %   
    %   [___, after_yn] = GPGA(___) returns the signal after the GPGA 
    %   process.
    %
    %   [___, isLarge] = GPGA(___) returns true if the current NMSE is 
    %   larger than the previous NMSE; otherwise, it returns false. (This
    %   check is not currently in use.)

    narginchk(5, 5);

    [sigSize, ~] = size(sigTarget);

    if floor(stopCriteria) == stopCriteria
        maxIter = stopCriteria;
    else
        maxIter = Inf;
    end
    
    % initial for GPGA
    sigRefFreq = 1 / sqrt(sigSize) .* fft(sigRef, sigSize, 1);
    restoringSig = dataMask .* sigTarget;
    currentIter = 0;
    
    if nargout > 1
        process = zeros([size(restoringSig) maxIter], 'like', sigTarget(1));
    end

    % GPGA process
    while maxIter > currentIter
        restoringSinFreq = 1 / sqrt(sigSize) .* fft(restoringSig, sigSize, 1);
        restoringSinFreq = (1 - transMask) .* restoringSinFreq + transMask .* sigRefFreq;
        restoringSig = sqrt(sigSize) .* ifft(restoringSinFreq, sigSize, 1);
        restoringSig = (1 - dataMask) .* restoringSig + dataMask .* sigTarget;

        currentIter = currentIter + 1;
        if nargout > 1
            process(:,:, currentIter) = restoringSig;
        end
    end
    restoredSig = restoringSig;

end
