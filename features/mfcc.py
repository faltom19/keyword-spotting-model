import numpy as np

SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP_SIZE = 256
NUM_FILTERS = 26
NUM_COEFFS = 13
MIN_FREQ = 20.0
MAX_FREQ = 8000.0
LOG_EPSILON = 1e-10
TWO_PI =  6.283185307179586

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz( mel):
    return 700 * (np.power(10,(mel / 2595)) - 1)

def hann_window(length):
    n = np.arange(length)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / length))

def mel_filterbank(num_bins):
    mel_min = hz_to_mel(MIN_FREQ)
    mel_max = hz_to_mel(MAX_FREQ)
    mel_points = np.linspace(mel_min, mel_max, NUM_FILTERS + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.round(hz_points * FFT_SIZE / SAMPLE_RATE).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)

    filterbank = np.zeros((NUM_FILTERS, num_bins))

    for f in range(0, NUM_FILTERS):
        left = bins[f]
        center = bins[f + 1]
        right = bins[f + 2]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        if(right > (num_bins - 1)):
            raise ValueError(f"filter {f} exceeds spectrum: right={right}, num_bins={num_bins}")

        for k in range(left, right + 1):
            if k < center:
                filterbank[f, k] = (k - left) / (center - left)
            elif k == center:
                filterbank[f, k] = 1.0
            else:
                filterbank[f, k] = (right - k) / (right - center)

    return filterbank

def dct_matrix(num_inputs, num_outputs):
    matrix_dct = np.zeros((num_outputs, num_inputs))

    for k in range(0, num_outputs):
        if(k == 0):
            alpha = np.sqrt(1 / num_inputs)
        else:
            alpha = np.sqrt(2 / num_inputs)

        for n in range(0, num_inputs):
            matrix_dct[k, n] = alpha * np.cos(np.pi * k * (n + 0.5) / num_inputs)

    return matrix_dct

_WINDOW = hann_window(FFT_SIZE)
_FILTERBANK = mel_filterbank(FFT_SIZE // 2 + 1)
_DCT = dct_matrix(NUM_FILTERS, NUM_COEFFS)


def frame_count(signal_length):
    if signal_length < FFT_SIZE:
        return 0
    return (signal_length - FFT_SIZE) // HOP_SIZE + 1


def extract_mfcc(samples):
    signal = np.asarray(samples, dtype=np.float64)
    num_frames = frame_count(len(signal))

    if num_frames == 0:
        raise ValueError(
            f"signal too short: {len(signal)} samples, need at least {FFT_SIZE}"
        )

    features = np.zeros((num_frames, NUM_COEFFS))

    for i in range(num_frames):
        start = i * HOP_SIZE
        frame = signal[start:start + FFT_SIZE]

        windowed = frame * _WINDOW

        spectrum = np.fft.rfft(windowed, n=FFT_SIZE)

        power = np.abs(spectrum) ** 2

        mel_energies = _FILTERBANK @ power

        log_energies = np.log(mel_energies + LOG_EPSILON)

        features[i] = _DCT @ log_energies

    return features

if __name__ == "__main__":
    # Tono a 1000 Hz, come nell'esempio C++
    n = np.arange(SAMPLE_RATE)
    tone = (8000 * np.sin(2 * np.pi * 1000 * n / SAMPLE_RATE)).astype(np.int16)

    features = extract_mfcc(tone)
    print("shape:", features.shape)
    print("first frame:", np.round(features[0], 2))