import os
import math
import librosa
import numpy as np
import json
import soundfile as sf

SAMPLE_RATE = 22050
JSON_PATH_CLEAN = "data-json/data-clean.json"
JSON_PATH_DIST = "data-json/data-dist.json"
AUDIO_PATH = "C:/Users/Denes Leal/rep-git/TG-python/audios/"
STANDARD_DURATION = 10


def save_features_as_json(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=1, track_duration=4):
    data_clean = {
        "mfcc": [],
        "spectr_mag": [],
        "spectr_ang": [],
        "spectr_freq": []
    }

    data_dist = {
        "mfcc": [],
        "spectr_mag": [],
        "spectr_ang": [],
        "spectr_freq": []
    }

    samples_per_track = SAMPLE_RATE * track_duration
    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            audio_type = dirpath.split("\\")[-1]
            print("\nProcessando: {}".format(audio_type))

            for f in filenames:
                print("percorrendo, file: " + f)

                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start:finish],
                                                sr=sample_rate,
                                                n_mfcc=num_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    fft = np.fft.fft(signal[start:finish])
                    magnitude = np.abs(fft)
                    frequency = np.linspace(0, sample_rate, len(magnitude))
                    mag = magnitude[:int(len(frequency) / 2)]
                    freq = frequency[:int(len(frequency) / 2)]
                    ang = np.angle(fft)

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        print(audio_type)
                        if audio_type == "clean":
                            data_clean["mfcc"].append(mfcc.tolist())
                            data_clean["spectr_mag"].append(list(mag))
                            data_clean["spectr_ang"].append(list(ang))
                            data_clean["spectr_freq"].append(list(freq))

                        else:
                            data_dist["mfcc"].append(mfcc.tolist())
                            data_dist["spectr_mag"].append(list(mag))
                            data_dist["spectr_ang"].append(list(ang))
                            data_dist["spectr_freq"].append(list(freq))

    with open(JSON_PATH_CLEAN, "w") as fp:
        json.dump(data_clean, fp, indent=4)

    with open(JSON_PATH_DIST, "w") as fp:
        json.dump(data_clean, fp, indent=4)


def prepare_samples(dataset_path, length=4):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            for f in filenames:
                print("percorrendo, file: " + f)

                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                duration = librosa.get_duration(y=signal, sr=sample_rate)
                segments = math.ceil(duration / length)
                samples_per_segment = sample_rate * length

                print("dados: (duration, segments, num_samples, samples per segment: " + str(duration) + " " + str(
                    segments) + " " + str(samples_per_segment))

                output_path = "audios\prepared_datasets\\" + dirpath.split("\\")[-1] + "\\"

                for d in range(segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    if finish < len(signal):
                        output_file_path = output_path + f + "-" + str(d) + ".wav"
                        print("montando arquivo: " + output_file_path)

                        sf.write(output_file_path, signal[start:finish], sample_rate, format='wav', subtype='PCM_24')


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


if __name__ == "__main__":
    a = load_audio('1.wav')
    b = load_audio('1D.wav', type='dist')

    a = np.ravel(a)
    b = np.ravel(b)

    b = remove_silences(a, b)
    output_audio(b, 'teste_remover_silencios')
