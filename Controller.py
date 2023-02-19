import DataPrepper
import ModelManager
import ClassifierNetwork


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





if __name__ == "__main__":
    prepare_classifier_network()

    #Próximos passos: criar método para efetuar o predict + desnormalização do resultado + exportar para áudio
    #Fazer também um script com métodos de gerar gráficos/fazer analise
    #Depois fazer e testar o autoencoder!