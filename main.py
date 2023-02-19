import pydub
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pydub.effects import speedup
from scipy.io import wavfile
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np

#Importar áudios para treino e teste

c1 = AudioSegment.from_file("audios/datasets/clean/1.wav")
#c2 = AudioSegment.from_file("audios/2.wav")
#c3 = AudioSegment.from_file("/audios/3.wav")
#c4 = AudioSegment.from_file("/audios/mp3/4.mp3")
#c5 = AudioSegment.from_file("/audios/mp3/5.mp3")
#c6 = AudioSegment.from_file("/audios/mp3/6.mp3")
#c7 = AudioSegment.from_file("/audios/mp3/7.mp3")
#c8 = AudioSegment.from_file("/audios/mp3/8.mp3")
#c9 = AudioSegment.from_file("/audios/mp3/9.mp3")
#c10 = AudioSegment.from_file("/audios/mp3/10.mp3")

d1 = AudioSegment.from_file("audios/datasets/dist/1D-S.wav")
#d2 = AudioSegment.from_file("audios/2D.wav")
#d3 = AudioSegment.from_file("/audios/3D.wav")
#d4 = AudioSegment.from_file("/audios/mp3/4D.mp3")
#d5 = AudioSegment.from_file("/audios/mp3/5D.mp3")
#d6 = AudioSegment.from_file("/audios/mp3/6D.mp3")
#d7 = AudioSegment.from_file("/audios/mp3/7D.mp3")
#d8 = AudioSegment.from_file("/audios/mp3/8D.mp3")
#d9 = AudioSegment.from_file("/audios/mp3/9D.mp3")
#d10 = AudioSegment.from_file("/audios/mp3/10D.mp3")

x_audio = c1 #+ c2# + c3# + c4 + c5 + c6 + c7 + c8 + c9 + c10
y_audio = d1 #+ d2# + d3# + d4 + d5 + d6 + d7 + d8 + d9 + d10

c11 = AudioSegment.from_file("audios/datasets/clean/2.wav")
#c12 = AudioSegment.from_file("audios/12.wav")
#c13 = AudioSegment.from_file("/audios/13.wav")
#c14 = AudioSegment.from_file("/audios/mp3/14.mp3")
#c15 = AudioSegment.from_file("/audios/mp3/15.mp3")
#c16 = AudioSegment.from_file("/audios/mp3/16.mp3")
#c17 = AudioSegment.from_file("/audios/mp3/17.mp3")
#c18 = AudioSegment.from_file("/audios/mp3/18.mp3")
#c19 = AudioSegment.from_file("/audios/mp3/19.mp3")

d11 = AudioSegment.from_file("audios/datasets/dist/2DS.wav")
#d12 = AudioSegment.from_file("audios/12D.wav")
#d13 = AudioSegment.from_file("/audios/13D.wav")
#d14 = AudioSegment.from_file("/audios/mp3/14D.mp3")
#d15 = AudioSegment.from_file("/audios/mp3/15D.mp3")
#d16 = AudioSegment.from_file("/audios/mp3/16D.mp3")
#d17 = AudioSegment.from_file("/audios/mp3/17D.mp3")
#d18 = AudioSegment.from_file("/audios/mp3/18D.mp3")
#d19 = AudioSegment.from_file("/audios/mp3/19D.mp3")

w_audio = c11 #+ c12# + c13# + c14 + c15 + c16 + c17 + c18 + c19
z_audio = d11 #+ d12# + d13# + d14 + d15 + d16 + d17 + d18 + d19]

#Declarações de funções auxiliares para o funcionamento simplificado do algorítimo

# 'Corta' as rebarbas dos sinais, para que tenham o mesmo tamanho
def cutsignal(sx, sy):
    lx = len(sx)
    ly = len(sy)
    left_over = abs(lx-ly)
    a = left_over//2
    b = max(lx,ly)-(left_over-a)
    if lx>ly:
        sx = sx[a:b]
    else:
        sy = sy[a:b]
    return sx, sy

# Transforma os dados obtidos do arquivo mp3 para modulação por códigos de pulsos (PCM)
def topcm(data):
    new = []
    for i in range(len(data)//2):
        sample = int.from_bytes(data[i:i+2], 'little', signed=True)
        new.append(sample)
    return new

# Normaliza os dados
def normalize(data):
    return data/max(data)

# cria um arquivo mp3 através dos dados obtidos pelo algorítimo
# crédito: https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3
def create_mp3(path, frame_rate, data, normalized=False):
    if data.ndim == 2 and data.shape[1] == 2:
        channels = 2
    else:
        channels = 1

    if normalized:  # Se os valores estão na faixa [-1, 1]
        y = np.int16(data * 2 ** 15)
    else:
        y = np.int16(data)
    audio = AudioSegment(y.tobytes(), frame_rate=frame_rate, sample_width=2, channels=channels)
    # audio = speedup(audio, 1, 150)
    # audio = changePitch(audio, 1.0)
    audio.export(path, format="wav")

def changePitch(audio, octaves):

    new_sample_rate = int(44100 * (2.0 ** octaves))

    # keep the same samples but tell the computer they ought to be played at the
    # new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
    hipitch_sound = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})

    # now we just convert it to a common sample rate (44.1k - standard audio CD) to
    # make sure it works in regular audio players. Other than potentially losing audio quality (if
    # you set it too low - 44.1k is plenty) this should now noticeable change how the audio sounds.
    return hipitch_sound.set_frame_rate(44100)

#Importação e tratamento dos dados

# Dados de Treino
#x_audio = AudioSegment.from_file("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Clean/1.wav")
#y_audio = AudioSegment.from_file("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Drive/1D.wav")

#x_rate, X = wavfile.read("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Clean/1.wav")
#y_rate, Y = wavfile.read("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Drive/1D.wav")

x_audio, y_audio = cutsignal(x_audio, y_audio)

x_data = x_audio._data
y_data = y_audio._data
X = topcm(x_data)
Y = topcm(y_data)
# X = x_data
# Y = y_data

# Dados de Teste
#w_audio = AudioSegment.from_file("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Clean/2.wav")
w_data = w_audio._data
#z_audio = AudioSegment.from_file("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Drive/2D.wav")
z_data = z_audio._data
W = topcm(w_data)
Z = topcm(z_data)
# W = w_data
# Z = z_data

#w_rate, W = wavfile.read("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Clean/2.wav")
#z_rate, Z = wavfile.read("/content/drive/MyDrive/UFABC/TG/TG/Python/Audios/Audios Drive/2D.wav")

# Todos os frame_rates são iguais
print(x_audio.frame_rate, y_audio.frame_rate, w_audio.frame_rate, z_audio.frame_rate)
frame_rate = 44100

#Normalização e padronização dos dados

# Dados de Treino
X = normalize(np.array(X))
Y = normalize(np.array(Y))
print(np.shape(X),np.shape(Y))
X = X.reshape(1,-1)
# X = np.reshape(X,1)
Y = Y.reshape(1,-1)
# Y = np.reshape(Y,1)
print(np.shape(X),np.shape(Y))

# # Dados de Teste
W = normalize(np.array(W))
Z = normalize(np.array(Z))
print(np.shape(W),np.shape(Z))
W = W.reshape(1,-1)
# W = np.reshape(W,1)
Z = Z.reshape(1,-1)
# Z = np.reshape(Z,1)
print(np.shape(W),np.shape(Z))

X, Y = cutsignal(X,Y)
W,Z = cutsignal(W,Z)

# Renomeação das variáveis
X_train, X_test, y_train, y_test = X, W, Y, Z

#Gráfico mostrando a diferença na distribuição de um arquivo sem efeito e com efeito
# plt.plot(Y, color='green')
# plt.savefig('graphs/distribuicao-sem-efeito.png', bbox_inches='tight')
# plt.plot(X, color='red')
# plt.savefig('graphs/distribuicao-com-efeito.png', bbox_inches='tight')

#Treinamento do algorítimo
print("Treinando algoritmo")
mlp = MLPRegressor(hidden_layer_sizes=(5), solver='adam', random_state=1)
mlp.fit(X_train, y_train)
print("Efetuando previsão")
y_pred = mlp.predict(X_test)

# Comparação da saída do algorítmo com o valor real
# plt.plot(y_test, color="red") # Valores reais
# plt.plot(y_pred, color='blue') # Valores preditos
# plt.savefig('graphs/comparacao-saida-valorreal.png', bbox_inches='tight')
print("R2 Score: ", r2_score(y_pred,y_test)) # R2 Score


#Exportação do resultado
print("Exportando audio")
create_mp3("audios/resultados/Rteste-novo-dado.wav", frame_rate, y_pred, normalized=True)