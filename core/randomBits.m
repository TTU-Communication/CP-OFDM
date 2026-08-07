function bits = randomBits(sz, options)
    arguments (Repeating)
        sz (1,:) double {mustBeInteger, mustBeNonnegative}
    end

    arguments
        options.OutputLocation (1,1) string ...
            {mustBeMember(options.OutputLocation, ...
            ["cpu", "gpu"])} = "cpu"

        options.DataType (1,1) string ...
            {mustBeMember(options.DataType, ...
            ["single", "double", "int8", "uint8", ...
            "int16", "uint16", "int32", "uint32", "logical"])} = "double"
    end

    if isempty(sz)
        error("randomBits:MissingSize", ...
            "Must define the size of the output array.");
    end

    if isscalar(sz)
        dims = sz{1};
    else
        % Using comma-separated syntax, each size argument must be scalar
        if ~all(cellfun(@isscalar, sz))
            error("randomBits:InvalidSizeSyntax", ...
                ["Using comma-separated syntax, each size argument must be scalar." ...
                 "Please use randomBits(2,3,4) or randomBits([2 3 4])."]);
        end

        dims = [sz{:}];
    end

    if options.OutputLocation == "gpu"
        bits = gpuArray.randi([0 1], dims, options.DataType);
    else
        bits = randi([0 1], dims, options.DataType);
    end
end