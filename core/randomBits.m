function bits = randomBits(sz)
    arguments (Repeating)
        sz (1,:) double {mustBeInteger, mustBeNonnegative}
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

    bits = randi([0 1], dims);
end