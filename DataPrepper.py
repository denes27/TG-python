import os
import math
import librosa
import numpy as np
import json

import AudioPrepper
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
        if log_spectrogram.min() != log_spectrogram.max():
            norm_spect = normalise(log_spectrogram)
            # Salvando espectrograma de forma (256, 862)
            if normalize is True:
                spect_array.append(norm_spect)
            else:
                spect_array.append(log_spectrogram)
    np.save(output_path, spect_array)
    return spect_array

def extract_normalized_spectrogram(signal):
    stft = librosa.stft(signal,
                        n_fft=STD_N_FFT,
                        hop_length=STD_HOP_LENGTH)[:-1]
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    # normalizar e salvar minmax (condicional para testar a normalização do output)
    norm_spect = normalise(log_spectrogram)
    return norm_spect


def load_spectrogram(label):
    pathX = 'data/spectrograms/clean/' + label + '.npy'
    pathY = 'data/spectrograms/dist/' + label + '.npy'
    pathZ = 'data/spectrograms/test/' + label + '.npy'
    minmaxpathx = 'minmax/inputs/' + label + '.npy'
    minmaxpathy = 'minmax/outputs/' + label + '.npy'
    minmaxpathz = 'minmax/test/' + label + '.npy'
    spectX = np.load(pathX)
    spectY = np.load(pathY)
    spectZ = np.load(pathZ)
    minmaxin = np.load(minmaxpathx)
    minmaxout = np.load(minmaxpathy)
    minmaxtest = np.load(minmaxpathz)
    return spectX, spectY, spectZ, minmaxin, minmaxout, minmaxtest


def load_audio_arrays(label='teste'):
    pathx = AUDIO_PATH + '/clean/' + label + '.npy'
    pathy = AUDIO_PATH + '/dist/' + label + '.npy'
    pathz = AUDIO_PATH + '/test/' + label + '.npy'

    datax = np.load(pathx)
    datay = np.load(pathy)
    dataz = np.load(pathz)

    return datax, datay, dataz

def normalise(array):
    if array.min() != array.max():
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (NORM_MAX - NORM_MIN) + NORM_MIN
        return norm_array


def denormalise(norm_array, original_min, original_max):
    if original_min != original_max:
        array = (norm_array - NORM_MIN) / (NORM_MAX - NORM_MIN)
        array = array * (original_max - original_min) + original_min
        return array


def save_min_max(inputs, outputs, tests, name='teste', normalize_outputs=False):
    minmaxin = []
    minmaxout = []
    minmaxtest = []
    save_path_in = 'minmax/inputs/' + name  # '.pk1'
    save_path_out = 'minmax/outputs/' + name  # '.pk1'
    save_path_test = 'minmax/test/' + name  # '.pk1'
    for x, y, z in zip(inputs, outputs, tests):
        minmaxin.append((inputs.min(), inputs.max()))
        minmaxtest.append((tests.min(), tests.max()))
        if normalize_outputs is True:
            minmaxout.append((outputs.min(), outputs.max()))
    np.save(save_path_in, minmaxin)
    np.save(save_path_out, minmaxout)
    np.save(save_path_test, minmaxtest)
    # with open(save_path, "wb") as f:
    #     pickle.dump(array, f)


def prepare_spectrogram_data(label='teste', normalize_output=False):
    x, y, test = AudioPrepper.load_audio_files()

    spectx = np.array(save_spectrogram(x, label, 'clean'))
    specty = np.array(save_spectrogram(y, label, 'dist', normalize=False))
    specttest = np.array(save_spectrogram(test, label, 'test'))

    save_min_max(spectx, specty, specttest, label, normalize_output)


def convert_spectrograms_to_audio(spectrograms, min_max_values, denormalize=False):
    signals = []
    print(np.array(spectrograms).shape)
    for spectrogram in spectrograms:
        # reshape the log spectrogram
        log_spectrogram = spectrogram[:, :, 0]
        # if denormalize is True:
        #     # apply denormalisation
        #     spectrogram = denormalise(
        #         log_spectrogram, min_max_value["min"], min_max_value["max"])

        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(log_spectrogram)
        # apply Griffin-Lim
        signal = librosa.istft(spec, hop_length=STD_HOP_LENGTH)
        # append signal to "signals"
        signals.append(signal)
    print("Forma após conversão: " + str(np.array(signals).shape))
    return np.ravel(signals)


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
