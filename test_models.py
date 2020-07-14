import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split

from attention_conv_noise_detection import build_attention_conv_network
from attention_noise_detection import build_attention_LSTM_network
from dataset import load_good_holter, load_dataset, load_mit
from dual_input_noise_detection import build_dual_input_network
from evaluation import train_eval, load_split
from generators import artefact_for_detection_dual, artefact_for_detection_3_in_2_out, artefact_for_detection
from recurrent_noise_detection import build_recurrent_network
from residual_noise_detection import build_residual_network
from simple_nn_noise_detection import build_simple_network
from triple_input_noise_detection import build_triple_detection_network
from unet_noise_detection import build_unet_1d

models = {"simple_detection": build_simple_network,
          "dense_detection": build_residual_network,
          "dual_detection": build_dual_input_network,
          "triple_detection": build_triple_detection_network,
          "unet_detection": build_unet_1d,
          "recurrent_detection": build_recurrent_network,
          "attention_LSTM_detection": build_attention_LSTM_network,
          "attention_conv_detection": build_attention_conv_network
          }

if __name__ == "__main__":
    for name in models:

        if name == "triple_detection":
            model = models[name]((4096, 1))
        else:
            model = models[name]((None, 1))

        #model = load_model("models\\" + name + "_paper_em.h5")
        model.summary()

        if name == "dual_detection":
            generator = artefact_for_detection_dual
        elif name == "triple_detection":
            generator = artefact_for_detection_3_in_2_out
        else:
            generator = artefact_for_detection

        res, _ = load_split()
        path = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\noise_lbl.csv"
        idxs = pd.read_csv(path, encoding='utf-8', sep=";")['i']
        X = load_dataset()["x"]
        x = X[np.where(idxs == 0, True, False), :, 0]
        x = np.expand_dims(x, 2)
        res1 = [0, 0]
        res1[0], res1[1] = train_test_split(x, test_size=0.25, random_state=42)

        res2 = load_good_holter()

        res3 = [0, 0]
        X = load_mit()
        res3[0], res3[1] = train_test_split(X, test_size=0.1, random_state=32)

        train_eval(model, (res1[0], res1[1]), only_eval=True, save_path=name + "_mit_test2", generator=generator,
                   size=4096, epochs=100,
                   noise_prob=[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 0], noise_type='em')
