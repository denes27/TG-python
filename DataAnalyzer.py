import matplotlib.pyplot as plt


def generate_graph(signal, title="graph"):
    plt.figure()
    plt.title(title)
    plt.plot(signal)
    plt.show()
