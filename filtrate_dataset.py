import pickle as pkl

import numpy as np

from evaluation import load_split
from generators import _generate_segment_artefact, autocorrelator


def count_clear(annotation):
    counter = 0
    cnt = 0
    cnter = [0] * 12
    for i in range(annotation.shape[0]):
        unique_pred, counts_pred = np.unique(annotation[i], return_counts=True)
        if np.max(annotation[i] == 4):
            counter += 1
            for j in reversed(range(12)):
                if counts_pred[-1] > j:
                    cnter[j] += 1
                    break
        else:
            cnt += 1
    print(counter, cnt, cnter)
    return counter, cnt, cnter


def save_cleared_dataset(x, y, pred, noise="em"):
    num1, num2, num = count_clear(pred)
    new_set1x = np.zeros((num2, x.shape[1], x.shape[2]))
    new_set1y = np.zeros((num2, y.shape[1]))
    counter = 0
    for i in range(pred.shape[0]):
        if np.max(pred[i]) != 4:
            new_set1x[counter, :, :] = x[i]
            new_set1y[counter, :] = y[i]
            counter += 1

    with open("data_denoised_" + noise + ".pkl", 'wb') as outfile:
        pkl.dump((new_set1x, new_set1y), outfile)


def predict_annotation(dataset, model, noise="em"):
    result = np.zeros((dataset.shape[0], dataset.shape[-1]))
    new_res = np.zeros((dataset.shape[0], 500, dataset.shape[-1]))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[-1]):
            new_res[i, :, j] = autocorrelator(dataset[i, :, j])[:500]
    print(dataset.shape, new_res.shape)
    for j in range(dataset.shape[-1]):
        y_val_cat_prob = model.predict([dataset[:, :, j:j + 1], dataset[:, ::16, j:j + 1], new_res[:, :, j:j + 1]])
        print(y_val_cat_prob[0].shape, y_val_cat_prob[1].shape)
        y_prediction = np.argmax(y_val_cat_prob[0], axis=-1)
        print(y_prediction.shape)
        new_pred = []
        for i in range(y_prediction.shape[0]):
            unique_pred, counts_pred = np.unique(y_prediction[i], return_counts=True)
            new_pred.append(unique_pred[np.argmax(counts_pred)])

        new_pred = np.array(new_pred)
        print(new_pred.shape)
        result[:, j] = new_pred
    print(result.shape)
    with open("pred_data_noised_" + noise + ".pkl", 'wb') as outfile:
        pkl.dump(result, outfile)


def make_noised_dataset(noise="em"):
    X, Y = load_split()
    res = X
    new_res = np.zeros(res[1].shape)
    for i in range(res[1].shape[0]):
        noise_lvl = np.random.choice(5, p=[0.3, 0.2, 0.1, 0.1, 0.2])
        for j in range(res[1].shape[-1]):

            new_res[i, :, j] = _generate_segment_artefact(res[1][i, :, j], noise_lvl, noise)

    with open("data_noised_" + noise + ".pkl", 'wb') as outfile:
        pkl.dump((new_res, Y[1]), outfile)


if __name__ == "__main__":
    type = "em"
    make_noised_dataset(type)

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from keras.engine.saving import load_model

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    set_session(tf.Session(config=config))

    with open("data_noised_" + type + ".pkl", 'rb') as infile:
        dataset = pkl.load(infile)

    model = load_model("models\\" + "triple_detection" + "_test1_" + type + ".h5")
    predict_annotation(dataset[0][:, :4096, :], model, type)
    # ''
    # with open("data_noised_ma.pkl", 'rb') as infile:
    #    dataset = pkl.load(infile)

    with open("pred_data_noised_" + type + ".pkl", 'rb') as infile:
        annotation = pkl.load(infile)

    save_cleared_dataset(dataset[0][:, :4096, :], dataset[1], annotation, type)

    with open("data_denoised_" + type + ".pkl", 'rb') as infile:
        annotation = pkl.load(infile)

    print(np.sum(annotation[1][:, 0]))
    print(np.sum(annotation[1][:, 123]))
