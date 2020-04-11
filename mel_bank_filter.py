from numpy import log, power, exp, zeros, linspace, round

def hz_to_mel(hz):
    return 1127*log(1+hz/700)

def mel_to_hz(mel):
    return 700*exp(mel/1127)-700

def make_mel_filter_bank(number_of_banks, fft_size, max_f, normalize=True):
    bank_filter = zeros((number_of_banks, fft_size))
    frequencies = linspace(0, hz_to_mel(max_f), number_of_banks+2)
    frequencies = mel_to_hz(frequencies)
    f_indexes = round((frequencies/max_f) * (fft_size-1)).astype('int')

    for i in range(0, number_of_banks):
        bank_filter[i, f_indexes[i]:f_indexes[i+1]+1] = linspace(0,1,f_indexes[i+1] - f_indexes[i]+2)[1:]
        bank_filter[i, f_indexes[i+1]+1:f_indexes[i+2]+1] = linspace(1,0,f_indexes[i+2] - f_indexes[i+1]+2)[1:-1]

        if normalize:
            bank_filter[i] = bank_filter[i] / sum(bank_filter[i])

    return bank_filter
