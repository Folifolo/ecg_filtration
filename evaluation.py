import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from generators import artefact_for_detection, DEFAULT_NOISE_PROB



def get_predict_labels(model, data):
    if type(data[1]) is list:
        y_prediction, y_labels = get_predict_labels_dual_out(model, data)
    else:
        y_prediction, y_labels = get_predict_labels_single_out(model, data)
    return y_prediction, y_labels


def get_predict_labels_single_out(model, data):
    X_test, Y_test = data

    y_val_cat_prob = model.predict(X_test)
    y_prediction = np.argmax(y_val_cat_prob, axis=-1)
    y_labels = np.argmax(Y_test, axis=-1)
    return y_prediction, y_labels


def get_predict_labels_dual_out(model, data):
    X_test, Y_test = data

    y_val_cat_prob = model.predict(X_test)
    y_prediction = np.argmax(y_val_cat_prob[0], axis=-1)
    y_labels = np.argmax(Y_test[0], axis=-1)
    return y_prediction, y_labels


def plot_results(y_prediction, y_labels, x):
    for i in range(len(y_prediction)):
        plt.subplot(7, 1, 1)
        plt.plot(x[0][i, :, 0])
        for j in np.arange(2, 8):
            x_axis = np.arange(y_prediction.shape[1])
            plt.subplot(7, 1, j)
            curr_pred = (y_prediction[i] == j - 2).astype(int)
            curr_label = (y_labels[i] == j - 2).astype(int)
            plt.fill_between(x_axis, [0.5] * y_prediction.shape[1], curr_label * 0.5 + 0.5, alpha=0.3, color='b')
            plt.fill_between(x_axis, [0] * y_prediction.shape[1], curr_pred * 0.5, alpha=0.3, color='r')

        plt.show()


def calculate_f1(y_prediction, y_labels):
    mean_f1 = 0
    classes = np.max(y_labels) + 1
    for i in np.arange(np.max(y_labels) + 1):
        curr_pred = y_prediction == i
        curr_label = y_labels == i
        if np.sum(curr_label) == 0:
            print("class " + str(i) + " is missing")
            classes -= 1
        else:
            curr_f1 = f1_score(curr_label, curr_pred, average="macro")
            print("F1 score for class " + str(i) + " == " + str(curr_f1.round(4)))
            mean_f1 += curr_f1

    print("Mean F1 score: " + str((mean_f1 / classes).round(4)))


def save_hist(h, name):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.savefig("pics\\" + name + ".png")
    plt.clf()


def plot_confusion(y_prediction, y_labels):
    import pandas as pd
    import seaborn as sn
    from sklearn.metrics import confusion_matrix
    y_prediction = y_prediction.flatten()
    y_labels = y_labels.flatten()
    matr = confusion_matrix(y_labels, y_prediction) / 4096/2000
    df_cm = pd.DataFrame(matr)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.show()


def new_metric(y_prediction, y_labels):
    accuracy = 0.0
    for i in range(y_prediction.shape[0]):
        for j in range(y_prediction.shape[1]):
            if (
                    (y_prediction[i, j] == y_labels[i, j]) or
                    (y_prediction[i, j] == y_labels[i, j] + 1) or
                    (y_prediction[i, j] == y_labels[i, j] - 1)):
                accuracy += 1.0
    accuracy /= y_prediction.shape[0]
    accuracy /= y_prediction.shape[1]
    print("modified accuracy " + str(accuracy))
    return  accuracy


def train_model(model, x, save_path="simple_detection", generator=artefact_for_detection,
                size=2048, epochs=150, noise_prob=None, noise_type='ma'):
    from keras.callbacks import ModelCheckpoint

    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB

    X_train = x[0]
    X_test = x[1]

    generator_test = generator(X_test, size, 500, noise_type=noise_type, noise_prob=noise_prob)
    val = next(generator_test)

    generator_train = generator(X_train, size, 10, noise_type='ma', noise_prob=noise_prob)

    model_path = "models\\" + save_path + "_" + noise_type + ".h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    h = model.fit_generator(generator_train, epochs=epochs, steps_per_epoch=20, validation_data=val,
                            callbacks=callbacks_list)

    save_hist(h, save_path)

    return model


def eval_model(model, x, generator=artefact_for_detection, size=2048, noise_prob=None, noise_type='ma'):
    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB

    X_test = x[1]

    generator_test = generator(X_test, size, 2000, noise_type=noise_type, noise_prob=noise_prob)
    val = next(generator_test)

    y_prediction, y_labels = get_predict_labels(model, val)

    calculate_f1(y_prediction, y_labels)
    new_metric(y_prediction, y_labels)
    binary_class(y_prediction, y_labels)

    #plot_confusion(y_prediction, y_labels)
    plot_results(y_prediction, y_labels, val[0])


def train_eval(model, x, only_eval=False, save_path="simple_detection_em", generator=artefact_for_detection,
               size=2048, epochs=150, noise_prob=None, noise_type='ma'):
    if not only_eval:
        model = train_model(model, x, save_path, generator, size, epochs, noise_prob, noise_type)
    eval_model(model, x, generator, size, noise_prob, noise_type)
    return model


def test_arc(model, x, generator=artefact_for_detection, size=2048, noise_prob=None, noise_type='ma'):
    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB

    X_test = x[1]

    generator_test = generator(X_test, size, 500, noise_type=noise_type, noise_prob=noise_prob)
    val = next(generator_test)

    X_test, Y_test = val
    y_val_cat_prob = model.predict(X_test)
    y_prediction = np.argmax(y_val_cat_prob, axis=-1)
    y_labels = np.argmax(Y_test, axis=-1)
    for i in range(len(X_test)):
        qqq = np.sum(np.abs(y_labels[i, :] - y_prediction[i, :])) / 5
        if qqq > 150:
            print(qqq)
            plt.subplot(211)
            plt.plot(X_test[i])

            plt.subplot(212)
            plt.plot(y_labels[i])
            plt.plot(y_prediction[i])
            plt.legend(["label", "pred"])

            plt.show()


def to_binary(y_labels):
    class54 = y_labels == 5
    class54 += y_labels == 4
    class54 += y_labels == 3
    res = np.zeros(y_labels.shape[0])
    for i in range(y_labels.shape[0]):
        if np.sum(class54[i]) > y_labels.shape[1] // 2:
            res[i] = 1
    return np.array(res)


def binary_class(y_prediction, y_labels):
    bin_labels = to_binary(y_labels)
    bin_pred = to_binary(y_prediction)

    curr_f1 = f1_score(bin_labels, bin_pred, average="macro")
    print("F1 score binary == " + str(curr_f1.round(4)))


def load_split():
    X = load_dataset()['x']
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
    return X_train, X_test
