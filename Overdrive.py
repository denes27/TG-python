import json
import os
import math
import librosa
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

DATASET_PATH = "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\dataset"
PREPARED_DATASET_PATH = "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\prepared_datasets"
JSON_PATH_CLEAN = "data-json\\data-clean.json"
JSON_PATH_DIST = "data-json\\data-dist.json"
RESULT_AUDIO_PATH= "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\resultados\\"
SAMPLE_RATE = 22050


def save_features(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments = 1, track_duration=4):

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
                    mag = magnitude[:int(len(frequency)/2)]
                    freq = frequency[:int(len(frequency)/2)]
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

                print("dados: (duration, segments, num_samples, samples per segment: " + str(duration) + " " + str(segments) + " " + str(samples_per_segment))

                output_path = "audios\prepared_datasets\\" + dirpath.split("\\")[-1] + "\\"

                for d in range(segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    if finish < len(signal):
                        output_file_path = output_path + f + "-" + str(d) + ".wav"
                        print("montando arquivo: " + output_file_path)

                        sf.write(output_file_path, signal[start:finish], sample_rate, format='wav', subtype='PCM_24')


def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    mfcc = np.array(data["mfcc"])
    mag = np.array(data["spectr_mag"])
    ang = np.array(data["spectr_ang"])
    freq = np.array(data["spectr_freq"])

    print("Data succesfully loaded!")

    return mfcc, mag, ang, freq

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def standardize(train, test):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)+0.000001

    X_train = (train - mean) / std
    X_test = (test - mean) /std
    return X_train, X_test


if __name__ == "__main__":
    #prepare_samples(DATASET_PATH)
    save_features(PREPARED_DATASET_PATH, num_mfcc=42)
    mfccc, magc, angc, freqc = load_data(JSON_PATH_CLEAN)
    mfccd, magd, angd, freqd = load_data(JSON_PATH_DIST)

    #Primeira tentativa: usando MFCCs
    x_train, x_test, y_train, y_test = train_test_split(mfccc, mfccd, test_size=0.3)
    x_train, x_test = standardize(x_train, x_test)
    #y_train, y_test = standardize(y_train, y_test)

    # x_train = x_train.reshape(4325, 13)
    # y_train = y_train.reshape(4325, 13)
    # x_test = x_test.reshape(1903, 13)
    # y_test = y_test.reshape(1903, 13)

    # Initializing Device Specification
    device_spec = tf.DeviceSpec(job="localhost", replica=0, device_type="GPU")

    # Printing the DeviceSpec
    #print('Device Spec: ', device_spec.to_string())

    # Enabling device logging
    tf.debugging.set_log_device_placement(False)
    print("xtrain shape: {}".format(x_train.shape))
    print("ytrain shape: {}".format(y_train.shape))
    print("xtest shape: {}".format(x_test.shape))
    print("ytest shape: {}".format(x_test.shape))
    #print("mfccc ex: {}".format(mfccc[0:20]))
    #print("mfccd ex: {}".format(mfccd[0:20]))

    # Specifying the device
    with tf.device(device_spec):
        # build network topology
        model = keras.Sequential([

            # input layer
            #keras.layers.Flatten(input_shape=(mfccc.shape[1], mfccc.shape[2])),

            # 1st dense layer
            keras.layers.Dense(512, input_shape=(tf.TensorShape([173, 42])), activation='relu'),

            # 2nd dense layer
            keras.layers.Dense(256, activation='relu'),

            # 3rd dense layer
            keras.layers.Dense(42, activation='relu'),

            # output layer
            #keras.layers.Dense(13, activation='softmax'),
            keras.layers.Reshape((173, 42))
        ])
        for layer in model.layers:
            print(layer.output_shape)

        # compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.001)
        model.compile(optimizer=optimiser,
                      loss="mse",
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)
        #plot_history(history)
        print("Gerando predição:")
        # prediction = model.predict(x_test[:100])
        # comparison = model.predict(y_test[:100])
        file_path = "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\prepared_datasets\\clean\\3.wav-5.wav"
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal,
                                    sr=sample_rate,
                                    n_mfcc=42,
                                    n_fft=2048, hop_length=512)
        print(mfcc.shape)
        comparisonaudio = librosa.feature.inverse.mfcc_to_audio(mfcc, n_mels=42, n_fft=2048, hop_length=512, dct_type=2, norm='ortho',
                                                                ref=1.0,
                                                                lifter=0)
        mfcc = mfcc.T
        print(mfcc.shape)
        mfcc = mfcc.reshape(-1, 173, 42)
        mfccd = model.predict(np.array(mfcc))
        print(mfccd.shape)
        mfccd = mfccd[::-1, ::-1].T
        print(mfccd.shape)
        mfccd = mfccd.reshape(42, 173)
        # print(prediction)
        # print(prediction.shape)
        print("Gerando audio de mfccs:")
        # outputaudio = librosa.feature.inverse.mfcc_to_audio(prediction, n_mels=13, dct_type=2, norm='ortho', ref=1.0, lifter=0)
        outputaudio = librosa.feature.inverse.mfcc_to_audio(mfccd, n_mels=42, n_fft=2048, hop_length=512, dct_type=2,
                                                            ref=1.0, lifter=0, norm='ortho')
        # print(outputaudio)
        # print(outputaudio.shape)
        print("Gerando arquivo wav:")
        nomearquivo = "resultado.wav"
        nomearquivo2 = "comparison.wav"
        output_file_path =RESULT_AUDIO_PATH + nomearquivo
        output_file_path2 =RESULT_AUDIO_PATH + nomearquivo2
        sf.write(output_file_path, np.ravel(outputaudio), SAMPLE_RATE, format='wav', subtype='PCM_24')
        sf.write(output_file_path2, np.ravel(comparisonaudio), SAMPLE_RATE, format='wav', subtype='PCM_24')
    # mlp = MLPRegressor(hidden_layer_sizes=(10), solver='adam', random_state=1)
    # mlp.fit(x_train, y_train.ravel())
    # y_pred = mlp.predict(x_test)
