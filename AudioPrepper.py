import os
import math
import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
AUDIO_PATH = "audios/"
STANDARD_DURATION = 10


def split_duration(signals):
    cut_signals = []
    for signal in signals:

        original_duration = librosa.get_duration(y=signal, sr=SAMPLE_RATE)
        segments = math.ceil(original_duration / STANDARD_DURATION)
        samples_per_segment = SAMPLE_RATE * STANDARD_DURATION

        print("dados: (duração original: {}, qtd segmentos: {}, samples por segmento: {}, total samples: {}".format(
            original_duration, segments, samples_per_segment, samples_per_segment * segments))

        for d in range(segments):
            start = samples_per_segment * d
            finish = start + samples_per_segment
            if finish <= len(signal):
                cut_signals.append(signal[start:finish])
    return cut_signals


def obtain_signals(dataset_path):
    signals_clean = []
    signals_dist = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            for f in filenames:
                print("percorrendo, file: " + f)

                file_path = os.path.join(dirpath, f)
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                current_directory = dirpath.split("\\")[-1]
                print(current_directory)
                if current_directory == 'clean':
                    signals_clean.append(signal)
                elif current_directory == 'dist':
                    signals_dist.append(signal)
    return signals_clean, signals_dist


def remove_silences(signal_a, signal_b):
    i = 0
    for sample_a in signal_a:
        if sample_a == 0:
            signal_b[i] = 0
        i += 1
    return signal_b


def load_audio(file_name, folder='datasets', type='clean'):
    # folder: datasets, prepared_datasets; type: clean, dist;
    if folder == 'prepared':
        folder = 'prepared_datasets'
    file_path = AUDIO_PATH + folder + '/' + type + '/' + file_name

    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return signal


def output_audio(signal, file_name, folder='resultados'):
    output_file_path = AUDIO_PATH + folder + '/' + file_name + '.wav'
    sf.write(output_file_path, signal, SAMPLE_RATE, format='wav', subtype='PCM_24')
    print('Áudio criado no caminho {}'.format(output_file_path))


def apply_padding(signal_a, signal_b):
    num_missing_items_a = STANDARD_DURATION * SAMPLE_RATE - len(signal_a)
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


def prepare_samples():
    print("Obtendo sinais a partir das amostras originais")
    sigs_clean, sigs_dist = obtain_signals(AUDIO_PATH + 'datasets')
    print(len(sigs_clean))
    print(len(sigs_dist))
    print("Dividindo sinais em amostras de {} segundos".format(STANDARD_DURATION))
    sigs_clean = split_duration(sigs_clean)
    sigs_dist = split_duration(sigs_dist)
    outputs_clean = []
    outputs_dist = []
    print("Removendo silencios e aplicando padding")
    for sig_a, sig_b in zip(sigs_clean, sigs_dist):
        # verificar se padding deve ser feito antes do split para garantir integridade das amostras
        sig_b = remove_silences(sig_a, sig_b)
        sig_a, sig_b = apply_padding(sig_a, sig_b)
        outputs_clean.append(sig_a)
        outputs_dist.append(sig_b)

    i = 0
    j = 0
    print("Exportando áudios tratados")
    for signal in outputs_clean:
        output_audio(signal, str(i), 'prepared_datasets/clean')
        i += 1

    for signal in outputs_dist:
        output_audio(signal, str(j) + 'D', 'prepared_datasets/dist')
        j += 1


if __name__ == "__main__":
    prepare_samples()
