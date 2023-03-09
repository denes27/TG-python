import os
import math
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050
AUDIO_PATH = "audios/"
PREPARED_AUDIO_PATH = "audios/prepared_datasets"
STANDARD_DURATION = 0.74
WAV_MAX = 32767


def split_duration(signals):
    cut_signals = []
    for signal in signals:

        original_duration = librosa.get_duration(y=signal, sr=SAMPLE_RATE)
        segments = math.ceil(original_duration / STANDARD_DURATION)
        samples_per_segment = SAMPLE_RATE * STANDARD_DURATION

        print("dados: (duração original: {}, qtd segmentos: {}, samples por segmento: {}, total samples: {}".format(
            original_duration, segments, samples_per_segment, samples_per_segment * segments))

        for d in range(segments):
            start = int(samples_per_segment * d)
            finish = int(start + samples_per_segment)
            if finish <= len(signal):
                cut_signals.append(signal[start:finish])
    return cut_signals


def obtain_signals(dataset_path):
    signals_clean = []
    signals_dist = []
    signals = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            for f in filenames:
                print("percorrendo, file: " + f)

                file_path = os.path.join(dirpath, f)
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                print(signal.shape)
                current_directory = dirpath.split("\\")[-1]
                print(current_directory)
                if current_directory == 'clean':
                    signals_clean.append(signal)
                elif current_directory == 'dist':
                    signals_dist.append(signal)
                else:
                    signals.append(signal)
    return np.array(signals_clean).ravel, np.array(signals_dist).ravel, np.array(signals).ravel


def load_audio_files():
    signals_x = []
    signals_y = []
    signals_test = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(PREPARED_AUDIO_PATH)):

        if dirpath is not AUDIO_PATH:
            folder = dirpath.split("\\")[-1]
            print("\nProcessando pasta: {}".format(folder))

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                if folder == 'clean':
                    signals_x.append(signal)
                elif folder == 'dist':
                    signals_y.append(signal)
                elif folder == 'test':
                    signals_test.append(signal)
            print("Pasta {} processada.".format(folder))
    return signals_x, signals_y, signals_test

def remove_silences(signal_a, signal_b):
    i = 0
    for sample_a in signal_a:
        if sample_a == 0:
            signal_b[i] = 0
        i += 1
    return signal_b


def load_audio(file_name, folder='datasets', type='clean', abs=False):
    # folder: datasets, prepared_datasets; type: clean, dist;
    if folder == 'prepared':
        folder = 'prepared_datasets'
    file_path = AUDIO_PATH + folder + '/' + type + '/' + file_name

    if abs is True:
        file_path = AUDIO_PATH + folder + '/' + file_name
    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return signal


def output_audio(signal, file_name, folder='resultados'):
    output_file_path = AUDIO_PATH + folder + '/' + file_name + '.wav'
    sf.write(output_file_path, signal, SAMPLE_RATE, format='wav', subtype='PCM_24')
    print('Áudio criado no caminho {}'.format(output_file_path))


def apply_padding(signal_a, signal_b):
    num_missing_items_a = math.ceil(STANDARD_DURATION * SAMPLE_RATE - len(signal_a))
    if num_missing_items_a > 0:
        signal_a = np.pad(signal_a,
                          (0, num_missing_items_a),
                          mode='constant')
    num_missing_items_b = len(signal_a) - len(signal_b)
    if num_missing_items_b > 0:
        signal_b = np.pad(signal_b,
                          (0, num_missing_items_b),
                          mode='constant')
    return signal_a, signal_b


def normalize_audio(signal):
    print("Normalizando sinal")
    #print(signal.shape)
    plt.figure(1)
    plt.title("signal")
    plt.plot(signal)
    plt.show()

    norm_signal = signal + WAV_MAX

    print("Plot 2")
    plt.figure(2)
    plt.title("added signal")
    plt.plot(norm_signal)
    plt.show()
    norm_signal = norm_signal / (2 * WAV_MAX)

    print("Plot 3")
    plt.figure(3)
    plt.title("norm signal")
    plt.plot(norm_signal)
    print("Mostrar")
    plt.show()

    return norm_signal


def denormalize_audio(signal):
    denorm_signal = signal * 2 * WAV_MAX
    return denorm_signal - WAV_MAX


def prepare_samples(path='datasets', output_path='prepared_datasets', normalized=True):
    print("Obtendo sinais a partir das amostras originais")
    sigs_clean, sigs_dist, sigs = obtain_signals(AUDIO_PATH + path)
    print("Dividindo sinais em amostras de {} segundos".format(STANDARD_DURATION))

    if normalized is True:
        sigs_clean = normalize_audio(sigs_clean)
        sigs_dist = normalize_audio(sigs_dist)
        sigs = normalize_audio(sigs)

    sigs_clean = split_duration(sigs_clean)
    sigs_dist = split_duration(sigs_dist)
    sigs = split_duration(sigs)
    outputs_clean = []
    outputs_dist = []
    outputs_sigs = []
    print("Removendo silencios e aplicando padding")
    for sig_a, sig_b, sigs in zip(sigs_clean, sigs_dist, sigs):
        # verificar se padding deve ser feito antes do split para garantir integridade das amostras
        sig_b = remove_silences(sig_a, sig_b)
        sig_a, sig_b = apply_padding(sig_a, sig_b)
        _ , sigs = apply_padding([], sigs)
        outputs_clean.append(sig_a)
        outputs_dist.append(sig_b)
        outputs_sigs.append(sigs)

    i = 0
    print("Exportando áudios tratados")
    for signal in outputs_clean:
        output_audio(signal, str(i), output_path + '/clean')
        i += 1

    i = 0
    for signal in outputs_dist:
        output_audio(signal, str(i) + 'D', output_path + '/dist')
        i += 1

    i = 0
    for signal in outputs_sigs:
        output_audio(signal, str(i), output_path + '/test')
        i += 1


if __name__ == "__main__":
    prepare_samples()
