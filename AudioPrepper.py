import os
import math
import librosa
import numpy as np
import soundfile as sf
import DataAnalyzer as da

SAMPLE_RATE = 22050
AUDIO_PATH = "audios/"
PREPARED_AUDIO_PATH = "audios/prepared_datasets"
STANDARD_DURATION = 0.74
WAV_MAX = 32767
LINUX_PATH_PREFIX = 'audios/datasets/'


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
                if current_directory == LINUX_PATH_PREFIX + 'clean':
                    print('Adicionando a sinais clean')
                    signals_clean.append(signal)
                    print(np.array(signals_clean).shape)
                elif current_directory == LINUX_PATH_PREFIX + 'dist':
                    print('Adicionando a sinais dist')
                    signals_dist.append(signal)
                    print(np.array(signals_dist).shape)
                else:
                    print('Adicionando a sinais test')
                    signals.append(signal)
                    print(np.array(signals).shape)

    print(np.array(signals_clean).shape)
    print(np.array(signals_dist).shape)
    print(np.array(signals).shape)
    # da.generate_graph(np.array(signals_clean).ravel(), "obtain signals: signals_clean")
    # da.generate_graph(np.array(signals_dist).ravel(), "obtain signals: signals_dist")
    # da.generate_graph(np.array(signals).ravel(), "obtain signals: signals_test")

    return signals_clean, signals_dist, signals


def load_audio_files():
    # Acho que esse processo não garante que os arquivos serão lidos na ordem alfabética, portanto pode ser que arquivos
    # sejam reconstruídos em ordem aleatoria
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


def normalize_audio(signals):
    print("Normalizando sinal")
    signals = np.array(signals)
    print(signals.shape)
    da.generate_graph(signals.ravel(), "normalize_audio: signal")

    norm_signals = []
    for signal in signals:
        signal = signal + WAV_MAX
        # da.generate_graph(norm_signal, "normalize_audio: added signal")

        signal = signal / (2 * WAV_MAX)
        norm_signals.append(signal)
        # norm_signal = (signal - np.mean(signal)) / np.std(signal)

    norm_signals = np.array(norm_signals)
    print(norm_signals.shape)
    da.generate_graph(norm_signals.ravel(), "normalize_audio: norm signal")

    return norm_signals


def denormalize_audio(signal):
    print("Desnormalizando sinal")
    print(signal.shape)
    signal = np.array(signal).ravel()

    da.generate_graph(signal, 'norm signal')
    # signal = signal - WAV_MAX
    # signal = signal - (signal.max()/2)
    # signal = signal - 0.10

    # da.generate_graph(signal, 'subtracted WAV_MAX signal')
    denorm_signal = signal * WAV_MAX * 2
    denorm_signal = denorm_signal - WAV_MAX

    da.generate_graph(denorm_signal, 'denorm signal')
    return denorm_signal


def save_audio_array(signals, label='teste', folder='clean', normalize=True):
    output_path = PREPARED_AUDIO_PATH + '/' + folder + '/' + label + '.npy'
    print(signals.shape)
    np.save(output_path, signals)
    return signals


def prepare_samples(path='datasets', output_path='prepared_datasets', normalized=True, split=True, pad=True):
    print("Obtendo sinais a partir das amostras originais")
    sigs_clean, sigs_dist, sigs = obtain_signals(AUDIO_PATH + path)

    if normalized is True:
        sigs_clean = normalize_audio(sigs_clean)
        # sigs_dist = normalize_audio(sigs_dist)
        sigs = normalize_audio(sigs)

    if split is True:
        print("Dividindo sinais em amostras de {} segundos".format(STANDARD_DURATION))
        sigs_clean = split_duration(sigs_clean)
        sigs_dist = split_duration(sigs_dist)
        sigs = split_duration(sigs)

    # da.generate_graph(np.array(sigs_clean).ravel(), "after split duration: signals_clean")
    # da.generate_graph(np.array(sigs_dist).ravel(), "after split duration: signals_dist")
    # da.generate_graph(np.array(sigs).ravel(), "after split duration: signals_test")

    outputs_clean = []
    outputs_dist = []
    outputs_sigs = []
    #print("Removendo silencios e aplicando padding")

    if pad is True:
        for sig_a, sig_b, sigs in zip(sigs_clean, sigs_dist, sigs):
            # verificar se padding deve ser feito antes do split para garantir integridade das amostras
            # sig_b = remove_silences(sig_a, sig_b) não posso remover silencios assim pq onda volta pro zero a cada periodo
            sig_a, sig_b = apply_padding(sig_a, sig_b)
            _, sigs = apply_padding([], sigs)
            outputs_clean.append(sig_a)
            outputs_dist.append(sig_b)
            outputs_sigs.append(sigs)

        # da.generate_graph(np.array(outputs_clean).ravel(), "after silence and padding: signals_clean")
        da.generate_graph(np.array(outputs_clean).ravel(), "output: signals_clean")
        # da.generate_graph(np.array(outputs_dist).ravel(), "after silence and padding: signals_dist")
        da.generate_graph(np.array(outputs_dist).ravel(), "output: signals_dist")
        # da.generate_graph(np.array(outputs_sigs).ravel(), "after silence and padding: signals_test")
        da.generate_graph(np.array(outputs_sigs).ravel(), "output: signals_test")

    i = 0
    print("Exportando dados dos audios tratados")
    save_audio_array(np.array(sigs_clean))
    save_audio_array(np.array(sigs_dist), folder='dist')
    save_audio_array(np.array(sigs), folder='test')
    # for signal in outputs_clean:
    #     output_audio(signal, str(i), output_path + '/clean')
    #     i += 1
    #
    # i = 0
    # for signal in outputs_dist:
    #     output_audio(signal, str(i) + 'D', output_path + '/dist')
    #     i += 1
    #
    # i = 0
    # for signal in outputs_sigs:
    #     output_audio(signal, str(i), output_path + '/test')
    #     i += 1


if __name__ == "__main__":
    prepare_samples(normalized=False, split=False, pad=False)
