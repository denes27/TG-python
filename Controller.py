import AudioPrepper
import DataPrepper
import ModelManager
import ClassifierNetwork
import VAE


def prepare_classifier_network(label='teste', save_model=False, model_name='model'):
    #Importar dados
    print('Iniciando preparação do classificador')
    inputs, outputs, _, _ = DataPrepper.load_spectrogram(label)
    #Importar modelo
    print('Gerando modelo')
    model = ClassifierNetwork.generateModel()
    #Retornar modelo treinado
    print('Treinando modelo')
    model = ModelManager.train_network(model, inputs, outputs)
    #Salvar modelo opcionalmente
    if save_model is True:
        print('Salvando modelo')
        ModelManager.save_model(model, model_name)
    print('Preparação concluída')
    return model

def prepare_vae_network():
    # Importar dados
    print('Iniciando preparação do classificador')
    inputs, outputs, _, _ = DataPrepper.load_spectrogram(label)
    # Importar modelo
    print('Gerando modelo')
    model = VAE.VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    # Retornar modelo treinado
    print('Treinando modelo')
    model = ModelManager.train_network(model, inputs, outputs)
    # Salvar modelo opcionalmente
    if save_model is True:
        print('Salvando modelo')
        ModelManager.save_model(model, model_name)
    print('Preparação concluída')
    return model

def obtain_result(model, data, minmax, file_name='teste', normalised_output=False):
    result = model.predict(data)
    if normalised_output is True:
        result = DataPrepper.denormalise(result, minmax[0], minmax[1])
    AudioPrepper.output_audio(result, file_name)



if __name__ == "__main__":
    model = prepare_classifier_network()
    inputs, outputs, _, _ = DataPrepper.load_spectrogram(teste)

    obtain_result(model, inputs[0])

    #Fazer também um script com métodos de gerar gráficos/fazer analise
    #Testar autoencoder