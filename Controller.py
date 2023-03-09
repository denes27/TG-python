import numpy as np
import matplotlib.pyplot as plt
import AudioPrepper
import DataPrepper
import ModelManager
import ClassifierNetwork
import VAE
import tensorflow as tf

LEARNING_RATE = 0.0005

def prepare_classifier_network(label='teste', save_model=False, model_name='model'):
    #Importar dados
    print('Iniciando preparação do classificador')
    inputs, outputs, _, _, _, _ = DataPrepper.load_spectrogram(label)
    #Importar modelo
    print('Gerando modelo')
    model = ClassifierNetwork.generateModel()
    #Retornar modelo treinado
    print('Treinando modelo')
    model, _ = ModelManager.train_network(model, inputs, outputs)
    #Salvar modelo opcionalmente
    if save_model is True:
        print('Salvando modelo')
        ModelManager.save_model(model, model_name)
    print('Preparação concluída')
    return model


def prepare_vae_network(label='teste', save_model=False, model_name='model_vae'):
    # Importar dados
    print('Iniciando preparação do classificador')
    # inputs, outputs, _, _, _, _ = DataPrepper.load_spectrogram(label)
    inputs, outputs, _ = AudioPrepper.load_audio_files()

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # inputs = np.reshape(inputs, (-1, 9, 1813, 1))
    inputs = np.reshape(inputs, (-1, 16317, 1, 1))
    # outputs = np.reshape(outputs, (-1, 9, 1813, 1))
    outputs = np.reshape(outputs, (-1, 16317, 1, 1))

    print(inputs.shape)
    print(outputs.shape)

    # Importar modelo
    print('Gerando modelo')
    vae = VAE.VAE(
        # input_shape=(9, 1813, 1),
        input_shape=(16317, 1, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        # conv_strides=(2, 2, 2, 2, (2, 1)),
        conv_strides=(1, 1, 1, 1, 1),
        latent_space_dim=128
    )
    vae.compile(LEARNING_RATE)
    vae.model.summary()
    # Retornar modelo treinado
    print('Treinando modelo')
    trained_vae, _ = ModelManager.train_vae(vae, inputs, outputs)
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
    _, _, test_data = AudioPrepper.load_audio_files()
    data = np.array(test_data)

    test_data = np.reshape(data, (-1, 16317, 1, 1))
    result = model.predict(test_data)

    print(result.shape)
    signal = np.ravel(result)
    print(signal.shape)

    plt.figure(1)
    plt.title("Test Data")
    plt.plot(data)

    plt.figure(2)
    plt.title("Prediction")
    plt.plot(signal)

    plt.show()

    # signal = AudioPrepper.denormalize_audio(signal)

    AudioPrepper.output_audio(signal, file_name)



if __name__ == "__main__":
    # model = prepare_classifier_network()
    # vae = prepare_vae_network(save_model=True, model_name='vae_audio_input_norm1')

    vae = VAE.VAE.load("models/vae_audio_input_norm1")


    model = vae.model
    #obtain_result(model, file_name='vaedelay1')
    obtain_result_audio_model(model, file_name='vaedrive_audio1')

    # Fazer também um script com métodos de gerar gráficos/fazer analise
    # Testar autoencoder