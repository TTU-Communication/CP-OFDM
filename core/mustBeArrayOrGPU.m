function mustBeArrayOrGPU(x)
%MUSTBEARRAYORGPU Summary of this function goes here
%   Detailed explanation goes here
    if ~(isnumeric(x) || isa(x, 'gpuArray'))
        eid = 'Input:notArrayOrGPU';
        msg = 'The input must be numeric or gpuArray.';
        throwAsCaller(MException(eid, msg));
    end

end