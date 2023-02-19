import json

import librosa
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = "/audios/datasets"
PREPARED_DATASET_PATH = "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\prepared_datasets"
JSON_PATH_CLEAN = "data-json\\data-clean.json"
JSON_PATH_DIST = "data-json\\data-dist.json"
RESULT_AUDIO_PATH= "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\resultados\\"
SAMPLE_RATE = 22050





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


def destandardize(input):
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)+0.000001

    output = (input + mean) * std

    return output


def model_spectrogram():
    mfccc, magc, angc, freqc = load_data(JSON_PATH_CLEAN)
    mfccd, magd, angd, freqd = load_data(JSON_PATH_DIST)

    x_train, x_test, y_train, y_test = train_test_split(mfccc, mfccd, test_size=0.3)


if __name__ == "__main__":
    #prepare_samples(DATASET_PATH)
    #save_features(PREPARED_DATASET_PATH, num_mfcc=42)
    mfccc, magc, angc, freqc = load_data(JSON_PATH_CLEAN)
    mfccd, magd, angd, freqd = load_data(JSON_PATH_DIST)

    #Primeira tentativa: usando MFCCs
    x_train, x_test, y_train, y_test = train_test_split(mfccc, mfccd, test_size=0.3)
    #x_train, x_test = standardize(x_train, x_test)
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
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=1000)
        model.save('saved-models/model-non-standard')
        #history = tf.keras.models.load_model('saved-models/model1')

        plot_history(history)
        print("Gerando predição:")
        # prediction = model.predict(x_test[:100])
        # comparison = model.predict(y_test[:100])
        file_path1 = "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\prepared_datasets\\clean\\1.wav-2.wav"
        file_path2 = "C:\\Users\\Denes Leal\\rep-git\\TG-python\\audios\\prepared_datasets\\dist\\1D.wav-2.wav"
        signal, sample_rate = librosa.load(file_path1, sr=SAMPLE_RATE)
        signal2, sample_rate2 = librosa.load(file_path2, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal,
                                    sr=sample_rate,
                                    n_mfcc=42,
                                    n_fft=2048, hop_length=512)
        mfcc2 = librosa.feature.mfcc(y=signal2,
                                    sr=sample_rate2,
                                    n_mfcc=42,
                                    n_fft=2048, hop_length=512)
        print(mfcc.shape)
        comparisonaudio = librosa.feature.inverse.mfcc_to_audio(mfcc2, n_mels=42, n_fft=2048, hop_length=512, dct_type=2, norm='ortho',
                                                                ref=1.0,
                                                                lifter=0)
        mfcc = mfcc.T
        print(mfcc.shape)
        mfcc = mfcc.reshape(-1, 173, 42)
        mfccd = model.predict(np.array(mfcc))
        #mfccd = destandardize(mfccd)
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
