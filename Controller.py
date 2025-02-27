import numpy as np
import matplotlib.pyplot as plt
import AudioPrepper
import DataPrepper
import ModelManager
import ClassifierNetwork
import VAE
import tensorflow as tf
import DataAnalyzer as da

# LEARNING_RATE = 0.000001
LEARNING_RATE = 0.000000115 # 1,15 * 10^-7
# LEARNING_RATE = 0.00000024
# LEARNING_RATE = 0.0000011
# LEARNING_RATE = 0.000003
DATA_SIZE = 33075


def prepare_classifier_network(label='teste', save_model=False, model_name='model'):
    # Importar dados
    print('Iniciando preparação do classificador')
    inputs, outputs, _, _, _, _ = DataPrepper.load_spectrogram(label)
    # Importar modelo
    print('Gerando modelo')
    model = ClassifierNetwork.generateModel()
    # Retornar modelo treinado
    print('Treinando modelo')
    model, _ = ModelManager.train_network(model, inputs, outputs)
    # Salvar modelo opcionalmente
    if save_model is True:
        print('Salvando modelo')
        ModelManager.save_model(model, model_name)
    print('Preparação concluída')
    return model


def prepare_vae_network(label='teste', save_model=False, model_name='model_vae'):
    # Importar dados
    print('Iniciando preparação do classificador')
    # inputs, outputs, _, _, _, _ = DataPrepper.load_spectrogram(label)
    # inputs, outputs, _ = AudioPrepper.load_audio_files()
    inputs, outputs, _ = DataPrepper.load_audio_arrays()

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # inputs = np.reshape(inputs, (-1, 9, 1813, 1))
    inputs = np.reshape(inputs, (-1, DATA_SIZE, 1, 1))
    # outputs = np.reshape(outputs, (-1, 9, 1813, 1))
    outputs = np.reshape(outputs, (-1, DATA_SIZE, 1, 1))

    print(inputs.shape)
    print(outputs.shape)

    # Importar modelo
    print('Gerando modelo')
    vae = VAE.VAE(
        # input_shape=(9, 1813, 1),
        input_shape=(DATA_SIZE, 1, 1),
        # input_shape=(16317, 1, 1),
        conv_filters=(512, 256, 128, 64, 32),
        # conv_filters=(1024, 512, 256, 128, 64),
        conv_kernels=(3, 3, 3, 3, 3),
        # conv_strides=(2, 2, 2, 2, (2, 1)),
        conv_strides=(1, 1, 1, 1, 1),
        latent_space_dim=128
        # latent_space_dim=256
    )

    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE * 100,
    #                                                                decay_steps=LEARNING_RATE,
    #                                                                decay_rate=LEARNING_RATE)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE)
        # lambda epoch: 1e-7 * 10 ** (epoch / 20))
    # tf.summary.scalar('learning_rate', learning_rate)
    vae.compile(LEARNING_RATE)
    vae.model.summary()
    # Retornar modelo treinado
    print('Treinando modelo')
    trained_vae, _ = ModelManager.train_vae(vae, inputs, outputs, lr_schedule)
    # Salvar modelo opcionalmente
    if save_model is True:
        print('Salvando modelo')
        trained_vae.save("models/" + model_name)
    print('Preparação concluída')
    return trained_vae


def obtain_result(model, label='teste', file_name='teste', normalised_output=False):
    _, _, test_data, _, _, test_minmax = DataPrepper.load_spectrogram(label)
    test_data = test_data[..., np.newaxis]
    result = model.predict(test_data)
    signal = DataPrepper.convert_spectrograms_to_audio(result, test_minmax)
    if normalised_output is True:
        signal = DataPrepper.denormalise(signal, test_minmax[0], test_minmax[1])
    print(np.array(signal).shape)
    AudioPrepper.output_audio(signal, file_name)


def obtain_result_audio_model(model, label='teste', file_name='teste', normalised_output=False):
    # _, _, test_data = AudioPrepper.load_audio_files()
    _, _, data = DataPrepper.load_audio_arrays()

    # da.generate_graph(data.ravel(), 'test data')

    test_data = np.reshape(data, (-1, DATA_SIZE, 1, 1))
    result = model.predict(test_data)

    print(result.shape)

    signal = result.squeeze().ravel()
    print(signal.shape)
    da.generate_graph(signal, 'Prediction')

    if normalised_output is True:
        signal = AudioPrepper.denormalize_audio(signal)

    print(signal.shape)
    AudioPrepper.output_audio(signal, file_name)


if __name__ == "__main__":
    # model = prepare_classifier_network()
    vae = prepare_vae_network(save_model=True, model_name='vae_midi_denorm_5')

    # vae = VAE.VAE.load("models/vae_midi_norm_1")

    model = vae.model
    # obtain_result(model, file_name='vaedelay1')
    # min loss dados c4: 283
    # min loss busca lr c3 - c5: 318,2561
    obtain_result_audio_model(model, file_name='vae_midi_denorm_5')
