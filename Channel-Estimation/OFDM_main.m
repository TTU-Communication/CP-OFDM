clc; clear;

%% parameter setting
FFT_size = 64;                          % FFT size
nullSCIdx = getNullSCIndex(FFT_size);   % Null Subcarrier Index
[pilotSCIdx, pilotValue] = getPilotSCIndexAndValue(FFT_size);   % Pilot Subcarrier Index & Value
CP_size = FFT_size * 1 / 4;             % Cyclic Prefix size
channel_length = 8;                     % Multipath length in rayleight distribution (no LoS)
constellation_symbols_amount = 16;      % The point amount of constellation
modulation_mode = 'QAM';                % Modulation (Avaliable with 'PSK', 'QAM')
base_signal_amount = 1000;              % testing signal numbers (will multiply a factor)
signals_per_transmit = 100;             % Every loop test signals
EbN0s = 0:1:20;                         % Energy per bit to noise power spectral density ratio(dB)

%% value depends on parameter
signal_size = FFT_size - length(nullSCIdx) - length(pilotSCIdx);    % Data subcarrier size
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

%% data storage
BER = zeros(1, length(EbN0s));
estBER = zeros(1, length(EbN0s));

%% CP-OFDM
tic
for EbN0_idx = 1:size(EbN0s, 2)
    % SNR calculation
    snr = EbN0s(EbN0_idx) + 10 * log10(bits_per_symbol) ...
        + 10 * log10(signal_size / (FFT_size + CP_size));
    % Calculate the amount of test signals based on SNR
    max_signal = (10 ^ floor(snr / 10)) * base_signal_amount;
    % BER storage depends on EbN0
    test_bers = zeros(1, max_signal / signals_per_transmit);
    estTestBers = zeros(1, max_signal / signals_per_transmit);

    fprintf('EbN0 = %2d, max signal number = %d\n', EbN0s(EbN0_idx), max_signal);

    parfor i = 1:(max_signal / signals_per_transmit)
        % Tx
        incoming_data_bits = RandomBits(bits_amount, signals_per_transmit);
        modulation_signal = Modulator(incoming_data_bits, constellation_symbols_amount);
        map_signal = subcarrierMapping(modulation_signal, FFT_size, pilotSCIdx, pilotValue, nullSCIdx);
        IFFT_signal = sqrt(FFT_size) .* ifft(map_signal, FFT_size);
        CP_signal = CPAdder(IFFT_signal, CP_size);

        % Channel
        signal_power = PowerCalculator(CP_signal);
        [channel_signal, channel] = channel_Rayleigh(CP_signal, channel_length, 1/channel_length, FFT_size);

        % Noise
        noise = noise_AWGN(size(channel_signal), snr, signal_power, channel_signal(1));
        noise_power = PowerCalculator(noise);
        channel_noise_signal = channel_signal + noise;

        % Rx
        remove_CP_signal = CPRemover(channel_noise_signal, CP_size);
        FFT_signal = 1 / sqrt(FFT_size) .* fft(remove_CP_signal, FFT_size);

        % actual channel
        EQ_signal = equalizer(FFT_signal, channel);
        demap_signal = subcarrierDemapping(EQ_signal, pilotSCIdx, nullSCIdx);
        output_data_bits = Demodulator(demap_signal, constellation_symbols_amount);

        % estimated channel
        channelEst = channelEstimation(FFT_signal, pilotSCIdx, pilotValue, nullSCIdx);
        estEQSig = equalizer(FFT_signal, channelEst);
        demap_signal = subcarrierDemapping(estEQSig, nullSCIdx, pilotSCIdx);
        estOutDataBits = Demodulator(demap_signal, constellation_symbols_amount);

        % BER calculate
        [~, test_bers(i)] = biterr(incoming_data_bits, output_data_bits);
        [~, estTestBers(i)] = biterr(incoming_data_bits, estOutDataBits);

    end

    BER(EbN0_idx) = mean(test_bers);
    estBER(EbN0_idx) = mean(estTestBers);

end
toc
%% plot BER
figure
semilogy(EbN0s, BER, DisplayName='actual channel');
hold on;
semilogy(EbN0s, estBER, DisplayName='estimated channel');
xlabel('$E_{b}/N_{0}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('BER', 'FontSize', 16);
legend;
grid on;
