import os
import math
import librosa
import numpy as np
import json
from AudioPrepper import SAMPLE_RATE
import pickle

AUDIO_PATH = "audios/prepared_datasets"
JSON_PATH_CLEAN = "data-json/data-clean.json"
JSON_PATH_DIST = "data-json/data-dist.json"
STD_N_FFT = 512
STD_HOP_LENGTH = 256
NORM_MAX = 1
NORM_MIN = 0


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


def save_spectrogram(signals, label='teste', folder='clean', normalize=True):
    spect_array = []
    output_path = 'data/spectrograms/' + folder + '/' + label + '.npy'
    for signal in signals:
        stft = librosa.stft(signal,
                            n_fft=STD_N_FFT,
                            hop_length=STD_HOP_LENGTH)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        # normalizar e salvar minmax (condicional para testar a normalização do output)
        norm_spect = normalise(log_spectrogram)
        # Salvando espectrograma de forma (256, 862)
        if normalize is True:
            spect_array.append(norm_spect)
        else:
            spect_array.append(log_spectrogram)
    np.save(output_path, spect_array)
    return spect_array


def load_spectrogram(label):
    pathX = 'data/spectrograms/clean/' + label + '.npy'
    pathY = 'data/spectrograms/dist/' + label + '.npy'
    minmaxpathx = 'minmax/inputs/' + label + '.npy'
    minmaxpathy = 'minmax/outputs/' + label + '.npy'
    spectX = np.load(pathX)
    spectY = np.load(pathY)
    minmaxin = np.load(minmaxpathx)
    minmaxout = np.load(minmaxpathy)
    return spectX, spectY, minmaxin, minmaxout


def load_audio_files():
    signals_x = []
    signals_y = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(AUDIO_PATH)):

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
            print("Pasta {} processada.".format(folder))
    return signals_x, signals_y


def normalise(array):
    norm_array = (array - array.min()) / (array.max() - array.min())
    norm_array = norm_array * (NORM_MAX - NORM_MIN) + NORM_MIN
    return norm_array


def denormalise(norm_array, original_min, original_max):
    array = (norm_array - NORM_MIN) / (NORM_MAX - NORM_MIN)
    array = array * (original_max - original_min) + original_min
    return array


def save_min_max(inputs, outputs, name='teste', normalize_outputs=False):
    minmaxin = []
    minmaxout = []
    save_path_in = 'minmax/inputs/' + name  # '.pk1'
    save_path_out = 'minmax/outputs/' + name  # '.pk1'
    for x, y in zip(inputs, outputs):
        minmaxin.append((inputs.min(), inputs.max()))
        if normalize_outputs is True:
            minmaxout.append((outputs.min(), outputs.max()))
    np.save(save_path_in, minmaxin)
    np.save(save_path_out, minmaxout)
    # with open(save_path, "wb") as f:
    #     pickle.dump(array, f)


def prepare_spectrogram_data(label='teste', normalize_output=False):
    x, y = load_audio_files()

    spectx = np.array(save_spectrogram(x, label, 'clean'))
    specty = np.array(save_spectrogram(y, label, 'dist'))

    save_min_max(spectx, specty, label, normalize_output)


if __name__ == "__main__":
    prepare_spectrogram_data()
    # a = np.load('data/spectrograms/clean/teste.npy')
    # b = np.load('data/spectrograms/dist/teste.npy')
    # c = np.load('minmax/inputs/teste.npy')
    # d = np.load('minmax/outputs/teste.npy')
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)
