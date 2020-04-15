from attention_conv_noise_detection import build_attention_conv_network
from attention_noise_detection import build_attention_LSTM_network
from evaluation import load_split, train_eval
from recurrent_noise_detection import build_recurrent_network
from residual_noise_detection import build_residual_network
from simple_nn_noise_detection import build_simple_network
from unet_noise_detection import build_unet_1d

models = {"simple_detection_em" : build_simple_network,
          "dense_detection_ma" : build_residual_network,
          "unet_detection" : build_unet_1d,
          "recurrent_detection" : build_recurrent_network,
          "attention_LSTM_detection" : build_attention_LSTM_network,
          "attention_conv_detection" : build_attention_conv_network
          }

if __name__ == "__main__":
    X = load_split()
    for name in models:
        model = models[name]((None, 1))
        model.summary()
        train_eval(model, X, only_eval=True, save_path=name, size=2048, epochs=150)
