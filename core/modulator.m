function outSig = modulator(inDataBits, modOrder, modType)
    arguments
        inDataBits {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty, mustBeMember(inDataBits, [0 1])}
        modOrder (1,1) {mustBeInteger, mustBePositive} = 2
        modType {mustBeMember(modType, {'psk', 'qam'})} = 'psk'
    end

    switch (modType)
        case 'psk'
            outSig = pskmod(inDataBits, modOrder , InputType="bit");
        case 'qam'
            outSig = qammod(inDataBits, modOrder, InputType="bit", UnitAveragePower=true);
        otherwise
            error('Modulator:invalidModulation', ...
                'The modulation type must be one of PSK or QAM.');
    end
end