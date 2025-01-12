clc; clear;

%% parameter setting
signal_size = 640;                  % Data subcarrier size
FFT_size = 2048;                    % FFT size
CP_size = FFT_size * 1 / 8;         % Cyclic Prefix size
% channel_length = 8;                 % Multipath length in rayleight distribution (no LoS)
constellation_symbols_amount = 16;  % The point amount of constellation
modulation_mode = 'QAM';            % Modulation (Avaliable with 'PSK', 'QAM')
base_signal_amount = 10000;         % testing signal numbers (will multiply a factor)
signals_per_transmit = 100;         % Every loop test signals
EbN0s = 0:1:20;                     % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
bits_per_symbol = log2(constellation_symbols_amount);
bits_amount = signal_size * bits_per_symbol;

%% package 
RandomBits = @(r, c) randi([0 1], r, c);
PowerCalculator = @(sig) sum(abs(sig) .^ 2) / size(sig, 1);
switch (lower(modulation_mode))
    case 'psk'
        Modulator = @(input, M) pskmod(input, M, InputType="bit");
        Demodulator = @(input, M) pskdemod(input, M, OutputType="bit");
    case 'qam'
        Modulator = @(input, M) qammod(input, M, InputType="bit", ...
            UnitAveragePower=true);
        Demodulator = @(input, M) qamdemod(input, M, OutputType="bit", ...
            UnitAveragePower=true);
    otherwise
        error('OFDMMain:invalidModulation', ...
            'The modulation mode must be one of PSK or QAM.');
end
CPAdder = @(sig, len) [sig(end-len+1:end, :); sig];
CPRemover = @(sig, len) sig(len+1:end, :);
PtoA_dB = @(sig) 10 * log10(max(abs(sig) .^ 2) ./ mean(abs(sig) .^ 2));

%% data storage
BER_CP = zeros(1, length(EbN0s));
BER_Companding = zeros(1, length(EbN0s));
CP_OFDM_sig_sample = [];
Companding_OFDM_sig_sample = [];

%% CP-OFDM
for EbN0_idx = 1:size(EbN0s, 2)
    % SNR calculation
    snr = EbN0s(EbN0_idx) + 10 * log10(bits_per_symbol) ...
        + 10 * log10(signal_size / (FFT_size + CP_size));
    % Calculate the amount of test signals based on SNR
    max_signal = (10 ^ floor(snr / 10)) * base_signal_amount;
    % BER storage depends on EbN0
    test_bers_cp = zeros(1, max_signal / signals_per_transmit);
    test_bers_companding = zeros(1, max_signal / signals_per_transmit);

    fprintf('EbN0 = %2d, max signal number = %d\n', EbN0s(EbN0_idx), max_signal);

    if EbN0_idx == length(EbN0s)
        CP_OFDM_PtoA_dB = zeros(max_signal / signals_per_transmit, signals_per_transmit);
        Companding_OFDM_PtoA_dB = zeros(max_signal / signals_per_transmit, signals_per_transmit);
        rand_idx = randi([1 max_signal / signals_per_transmit], 1);
    end

    parfor i = 1:(max_signal / signals_per_transmit)
        % Tx
        incoming_data_bits = RandomBits(bits_amount, signals_per_transmit);
        modulation_signal = Modulator(incoming_data_bits, constellation_symbols_amount);
        mapped_signal = mapping_subcarrier(modulation_signal, FFT_size);
        IFFT_signal = sqrt(FFT_size) .* ifft(mapped_signal, FFT_size);
        CP_signal = CPAdder(IFFT_signal, CP_size);
        Companding_signal = CP_signal;

        % Channel CP-OFDM
        signal_cp_power = PowerCalculator(CP_signal);
        % [channel_cp_signal, channel_cp] = channel_Rayleigh(CP_signal, ...
        %     channel_length, 1/channel_length, FFT_size);
        channel_cp_signal = CP_signal;
        % Channel Companding OFDM
        signal_companding_power = PowerCalculator(Companding_signal);
        % [channel_companding_signal, channel_companding] = channel_Rayleigh(Companding_signal, ...
        %     channel_length, 1/channel_length, FFT_size);
        channel_companding_signal = Companding_signal;

        % Noise CP-OFDM
        noise_cp = noise_AWGN(size(channel_cp_signal), snr, signal_cp_power, channel_cp_signal(1));
        noise_cp_power = PowerCalculator(noise_cp);
        channel_noise_cp_signal = channel_cp_signal + noise_cp;
        % Noise Companding OFDM
        noise_companding = noise_AWGN(size(channel_companding_signal), snr, ...
            signal_companding_power, channel_companding_signal(1));
        noise_companding_power = PowerCalculator(noise_companding);
        channel_noise_companding_signal = channel_companding_signal + noise_companding;

        % Rx CP-OFDM
        remove_CP_cp_signal = CPRemover(channel_noise_cp_signal, CP_size);
        FFT_cp_signal = 1 / sqrt(FFT_size) .* fft(remove_CP_cp_signal, FFT_size);
        % EQ_cp_signal = equalizer(FFT_cp_signal, channel_cp);
        EQ_cp_signal = FFT_cp_signal;
        demapped_cp_signal = demapping_subcarrier(EQ_cp_signal, signal_size);
        output_data_bits_cp = Demodulator(demapped_cp_signal, constellation_symbols_amount);
        % Rx Companding
        Decompanding_companding_signal = channel_noise_companding_signal;
        remove_CP_companding_signal = CPRemover(Decompanding_companding_signal, CP_size);
        FFT_companding_signal = 1 / sqrt(FFT_size) .* fft(remove_CP_companding_signal, FFT_size);
        % EQ_companding_signal = equalizer(FFT_companding_signal, channel_companding);
        EQ_companding_signal = FFT_companding_signal;
        demapped_companding_signal = demapping_subcarrier(EQ_companding_signal, signal_size);
        output_data_bits_companding = Demodulator(demapped_companding_signal, constellation_symbols_amount);


        % BER calculate
        [~, test_bers_cp(i)] = biterr(incoming_data_bits, output_data_bits_cp);
        [~, test_bers_companding(i)] = biterr(incoming_data_bits, output_data_bits_companding);

        if EbN0_idx == length(EbN0s)
            CP_OFDM_PtoA_dB(i, :) = PtoA_dB(CP_signal);
            Companding_OFDM_PtoA_dB(i, :) = PtoA_dB(Companding_signal);
            if i == rand_idx
                CP_OFDM_sig_sample = [CP_OFDM_sig_sample; CP_signal(:, 1)];
                Companding_OFDM_sig_sample = [Companding_OFDM_sig_sample; ...
                    Companding_signal(:, 1)];
            end

        end
        
    end

    BER_CP(EbN0_idx) = mean(test_bers_cp);
    BER_Companding(EbN0_idx) = mean(test_bers_companding);

end

%% plot BER
figure
semilogy(EbN0s, BER_CP, DisplayName='CP OFDM');
hold on;
semilogy(EbN0s, BER_Companding, DisplayName='Companding OFDM');
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
grid on;
legend;

%% plot PAPR
CP_OFDM_PtoA_dB = reshape(CP_OFDM_PtoA_dB, [], 1);
Companding_OFDM_PtoA_dB = reshape(Companding_OFDM_PtoA_dB, [], 1);

figure
[ECDF_CP, PAPR_CP] = ecdf(CP_OFDM_PtoA_dB);
CCDF_CP = 1 - ECDF_CP;
semilogy(PAPR_CP, CCDF_CP, DisplayName='CP OFDM');
grid on; hold on;
[ECDF_Companding, PAPR_Companding] = ecdf(Companding_OFDM_PtoA_dB);
CCDF_Companding = 1 - ECDF_Companding;
semilogy(PAPR_Companding, CCDF_Companding, DisplayName='Companding OFDM');
legend;
xlim([0 14]);
xlabel('Power (dB)');
ylabel('Probability');
title('PAPR');

%% plot PSD
figure
pspectrum(CP_OFDM_sig_sample);
grid on; hold on;
pspectrum(Companding_OFDM_sig_sample);
legend('CP OFDM', 'Companding OFDM');
