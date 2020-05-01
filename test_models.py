from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from attention_conv_noise_detection import build_attention_conv_network
from attention_noise_detection import build_attention_LSTM_network
from dataset import load_good_holter, load_dataset
from dual_input_noise_detection import build_dual_input_network
from evaluation import train_eval, load_split
from generators import artefact_for_detection_dual, artefact_for_detection_3_in_2_out, artefact_for_detection
from recurrent_noise_detection import build_recurrent_network
from residual_noise_detection import build_residual_network
from simple_nn_noise_detection import build_simple_network
from triple_input_noise_detection import build_triple_detection_network
from unet_noise_detection import build_unet_1d
import pandas as pd
import matplotlib.pyplot as plt

models = {"simple_detection": build_simple_network,
          "dense_detection": build_residual_network,
          "dual_detection": build_dual_input_network,
          "triple_detection": build_triple_detection_network,
          #"unet_detection": build_unet_1d,
          # "recurrent_detection": build_recurrent_network,
          # "attention_LSTM_detection": build_attention_LSTM_network,
          # "attention_conv_detection": build_attention_conv_network
          }

if __name__ == "__main__":
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    set_session(tf.Session(config=config))


    for name in models:

        if name == "triple_detection":
            model = models[name]((4096, 1))
        else:
            model = models[name]((None, 1))

        model = load_model("models\\" + name + "_p_ma.h5")
        # model.summary()

        if name == "dual_detection":
            generator = artefact_for_detection_dual
        elif name == "triple_detection":
            generator = artefact_for_detection_3_in_2_out
        else:
            generator = artefact_for_detection

        X = load_split()
        res = X
        path = "C:\\Users\\admin\\PycharmProjects\\noise\\noise_lbl.csv"
        idxs = pd.read_csv(path, encoding='utf-8', sep=";")['i']
        xy = load_dataset()
        X = xy["x"]
        x = X[np.where(idxs == 0, True, False), :, 0]
        x = np.expand_dims(x, 2)
        res1 = [0, 0]
        res1[0], res1[1] = train_test_split(x, test_size=0.25, random_state=42)

        print()
        print(name)
        print()
        res2 = load_good_holter()
        train_eval(model, (res1[0], res1[1]), only_eval=True, save_path=name+"_p", generator=generator, size=4096, epochs=50,
                   noise_prob=[1/5,1/5,1/5,1/5,1/5,0])
