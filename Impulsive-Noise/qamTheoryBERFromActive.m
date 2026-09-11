function ber = qamTheoryBERFromActive(snr, M)
    %QAMTHEORYBER Theoretical QAM BER from OFDM time-domain SNR
    %
    % snr   : Average time-domain sample SNR [dB]
    % M     : QAM modulation order
    % Nfft  : Actual transmitted IFFT size
    %         (including oversampling)
    % Nused : Number of occupied subcarriers
    %
    % Assumptions:
    %   - Unitary FFT/IFFT
    %   - AWGN
    %   - Gray-coded QAM
    %   - Noise power is defined relative to actual
    %     time-domain average signal power

    arguments
        snr {mustBeNumeric, mustBeReal}
        M (1,1) double {mustBeInteger, mustBePositive}
    end

    k = log2(M);

    if mod(k, 1) ~= 0
        error("M must be a power of 2.");
    end

    % Time-domain average SNR
    %              ↓
    % Data-subcarrier Es/N0
    % Es/N0 -> Eb/N0
    EbNo = snr - 10*log10(k);

    % Gray-coded QAM theoretical BER
    ber = berawgn(EbNo, "qam", M);
end