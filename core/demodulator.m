function outDataBits = demodulator(inSig, modOrder, modType)
    arguments
        inSig {mustBeArrayOrGPU, mustBeFinite, mustBeNonempty}
        modOrder (1,1) {mustBeInteger, mustBePositive} = 2
        modType {mustBeMember(modType, {'psk', 'qam'})} = 'psk'
    end

    switch (modType)
        case 'psk'
            outDataBits = pskdemod(inSig, modOrder, OutputType="bit");
        case 'qam'
            outDataBits = qamdemod(inSig, modOrder, OutputType="bit", UnitAveragePower=true);
        otherwise
            error('Demodulator:invalidModulation', 'The modulation type must be one of PSK or QAM.');
    end
end