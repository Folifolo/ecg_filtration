import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from generators import artefact_for_detection


def plot_results(model, data):
    y_val_cat_prob = model.predict(data[0])
    y_prediction = np.argmax(y_val_cat_prob, axis=-1)
    y_labels = np.argmax(data[1], axis=-1)
    for i in range(10):
        plt.subplot(7, 1, 1)
        plt.plot(data[0][i, :, 0])
        for j in np.arange(2, 8):
            x_axis = np.arange(y_prediction.shape[1])
            plt.subplot(7, 1, j)
            curr_pred = (y_prediction[i] == j - 2).astype(int)
            curr_label = (y_labels[i] == j - 2).astype(int)
            plt.fill_between(x_axis, [0.5] * y_prediction.shape[1], curr_label * 0.5 + 0.5, alpha=0.3, color='b')
            plt.fill_between(x_axis, [0] * y_prediction.shape[1], curr_pred * 0.5, alpha=0.3, color='r')

        plt.show()


def evaluate(model, data):
    X_test, Y_test = data
    mean_f1 = 0
    y_val_cat_prob = model.predict(X_test)
    y_prediction = np.argmax(y_val_cat_prob, axis=-1)
    y_labels = np.argmax(Y_test, axis=-1)
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
    plt.savefig("pics\\"+name+".png")
    plt.clf()


def train_model(model, x, save_path="simple_detection", generator=artefact_for_detection,
                size=2048, epochs=150, noise_prob=None, noise_type='ma'):
    from keras.callbacks import ModelCheckpoint

    if noise_prob is None:
        noise_prob = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5]

    X_train = x[0]
    X_test = x[1]

    generator_test = generator(X_test, size, 500, noise_type=noise_type, noise_prob=noise_prob)
    val = next(generator_test)

    generator_train = generator(X_train, size, 10, noise_type='ma', noise_prob=noise_prob)

    model_path = "models\\" + save_path + "_" + noise_type + ".h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    h = model.fit_generator(generator_train, epochs=epochs, steps_per_epoch=20, validation_data=val,
                        callbacks=callbacks_list, verbose=1)

    save_hist(h, save_path)

    return model


def eval_model(model, x, generator=artefact_for_detection,
               size=2048, noise_prob=None, noise_type='ma'):
    if noise_prob is None:
        noise_prob = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5]

    X_test = x[1]

    generator_test = generator(X_test, size, 500, noise_type=noise_type, noise_prob=noise_prob)
    val = next(generator_test)

    evaluate(model, val)
    plot_results(model, val)


def train_eval(model, x, only_eval=False, save_path="simple_detection_em", generator=artefact_for_detection,
               size=2048, epochs=150, noise_prob=None, noise_type='ma'):
    if not only_eval:
        model = train_model(model, x, save_path, generator, size, epochs, noise_prob, noise_type)
    eval_model(model, x, generator, size, noise_prob, noise_type)


def load_split():
    X = load_dataset()['x']
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
    return X_train, X_test
