import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
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
    nn = ["0", "1", "2", "3", "4"]
    for i in range(len(y_prediction)):
        plt.subplot(6, 1, 1)
        plt.plot(x[0][i, :, 0])
        plt.ylabel('ЭКГ')
        plt.xticks([])
        plt.yticks([])
        for j in np.arange(2, 7):
            x_axis = np.arange(y_prediction.shape[1])
            plt.subplot(6, 1, j)
            plt.ylabel(nn[j - 2])
            plt.xticks([])
            plt.yticks([])
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

            tn, fp, fn, tp = confusion_matrix(curr_label, curr_pred).ravel()

            test_se = tp / (tp + fn)
            test_sp = tn / (tn + fp)
            print("Val. Se = %s, Val. Sp = %s" % (round(test_sp, 4), round(test_se, 4)))
            print("accuracy ", accuracy_score(curr_label, curr_pred))
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
    class_names = ["0", "1", "2", "3", "4"]
    y_prediction = y_prediction.flatten()
    y_labels = y_labels.flatten()
    matr = confusion_matrix(y_labels, y_prediction) / 4096 / 2000
    df_cm = pd.DataFrame(matr, index=[i for i in class_names],
                         columns=[i for i in class_names])
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
    return accuracy


def new_metric_binary(y_prediction, y_labels):
    accuracy = 0.0
    for i in range(y_prediction.shape[0]):
        if (
                (y_prediction[i] == y_labels[i]) or
                (y_prediction[i] == y_labels[i] + 1) or
                (y_prediction[i] == y_labels[i] - 1)):
            accuracy += 1.0
    accuracy /= y_prediction.shape[0]
    print("modified accuracy " + str(accuracy))
    return accuracy


def train_model(model, x, save_path="simple_detection", generator=artefact_for_detection,
                size=2048, epochs=150, noise_prob=None, noise_type='ma', interv=10):
    from keras.callbacks import ModelCheckpoint

    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB

    X_train = x[0]
    X_test = x[1]

    generator_test = generator(X_test, size, 500, noise_type=noise_type, noise_prob=noise_prob, num_sections=interv)
    val = next(generator_test)

    generator_train = generator(X_train, size, 10, noise_type='ma', noise_prob=noise_prob, num_sections=interv)

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

    import time
    start = time.time()
    y_prediction, y_labels = get_predict_labels(model, val)
    done = time.time()
    elapsed = done - start
    print(elapsed)

    # calculate_f1(y_prediction, y_labels)
    # new_metric(y_prediction, y_labels)
    # binary_class(y_prediction, y_labels)

    # plot_confusion(y_prediction, y_labels)
    plot_roc(model, val)
    plot_confusion(y_prediction, y_labels)
    plot_results(y_prediction, y_labels, val[0])


def plot_roc(model, val):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    y_val_cat_prob = model.predict(val[0])

    pred = val[1][0][:, -2].flatten()
    lab = y_val_cat_prob[0][:, -2].flatten()
    fpr, tpr, thresholds = roc_curve(pred, lab)

    print("AUC = ", roc_auc_score(val[1][0].flatten(), y_val_cat_prob[0].flatten()))
    plt.plot(fpr, tpr)
    plt.show()


def eval_model_binary(model, x, generator=artefact_for_detection, size=2048, noise_prob=None, noise_type='ma'):
    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB

    X_test = x[1]

    generator_test = generator(X_test, size, 2000, noise_type=noise_type, noise_prob=noise_prob, num_sections=0)
    val = next(generator_test)
    """
    y_prediction, y_labels = get_predict_labels(model, val)
    new_pred = []
    new_labels = []
    for i in range(y_prediction.shape[0]):
        unique_pred, counts_pred = np.unique(y_prediction[i], return_counts=True)
        unique_labels, counts_labels = np.unique(y_labels[i], return_counts=True)
        new_labels.append(unique_labels[np.argmax(counts_labels)])
        new_pred.append(unique_pred[np.argmax(counts_pred)])
        
    new_labels = np.array(new_labels)
    new_pred = np.array(new_pred)
        
    calculate_f1(new_pred, new_labels)
    new_metric_binary(new_pred, new_labels)
    """
    plot_roc(model, val)

    # plot_confusion(y_prediction, y_labels)
    # plot_results(y_prediction, y_labels, val[0])


def train_eval(model, x, only_eval=False, save_path="simple_detection_em", generator=artefact_for_detection,
               size=2048, epochs=150, noise_prob=None, noise_type='ma'):
    if not only_eval:
        model = train_model(model, x, save_path, generator, size, epochs, noise_prob, noise_type, interv=10)
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
    # class54 += y_labels == 3
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
    xy = load_dataset()
    X = xy['x']
    Y = xy['y']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    # X_val, X_test = train_test_split(X_test, test_size=0.25, random_state=42)
    return (X_train, X_test), (Y_train, Y_test)
