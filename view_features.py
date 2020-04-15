import matplotlib.pyplot as plt
from see_rnn import get_layer_outputs, show_features_1D


def view_features(model, data, layer_num):
    outs = get_layer_outputs(model, data, layer_idx=layer_num)
    for i in range(len(data)):
        plt.plot(data[i])
        plt.show()
        show_features_1D(outs[i:i + 1], n_rows=8, show_borders=False)
