import json
import os
import pickle as pkl
import sys

import BaselineWanderRemoval as bwr
import numpy as np

DATA_PATH = "C:\\data\\"
DATA_FILENAME = "data_2033.json"
DIAG_FILENAME = "diagnosis.json"
PKL_FILENAME = "data_2033.pkl"
HOLTER_PATH = "holters\\"
HOLTER_FILENAME = "holter"
MIT_PATH = "mit\\mit_dataset.pkl"

LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 500


def parser(path):
    try:
        infile = open(path + DATA_FILENAME, 'rb')
        data = json.load(infile)
        diag_dict = get_diag_dict()

        X = []
        Y = []
        for id in data.keys():

            leads = data[id]['Leads']
            diagnosis = data[id]['StructuredDiagnosisDoc']

            y = []
            try:
                for diag in diag_dict.keys():
                    y.append(diagnosis[diag])
            except KeyError:
                print("\nThe patient " + id + " is not included in the final dataset. Reason: no diagnosis.")
                continue
            y = np.where(y, 1, 0)

            x = []
            try:
                for lead in LEADS_NAMES:
                    rate = int(leads[lead]['SampleRate'] / FREQUENCY_OF_DATASET)
                    x.append(leads[lead]['Signal'][::rate])
            except KeyError:
                print("\nThe patient " + id + " is not included in the final dataset. Reason: no lead.")
                continue

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        X = np.swapaxes(X, 1, 2)

        print("The dataset is parsed.")
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)

        return {"x": X, "y": Y}

    except FileNotFoundError:
        print("File " + DATA_FILENAME + " has not found.\nThe specified folder (" + path +
              ") must contain files with data (" + DATA_FILENAME +
              ") and file with structure of diagnosis (" + DIAG_FILENAME + ").")
        sys.exit(0)


def get_diag_dict():
    def deep(data, diag_list):
        for diag in data:
            if diag['type'] == 'diagnosis':
                diag_list.append(diag['name'])
            else:
                deep(diag['value'], diag_list)

    try:
        infile = open(DATA_PATH + DIAG_FILENAME, 'rb')
        data = json.load(infile)

        diag_list = []
        deep(data, diag_list)

        diag_num = list(range(len(diag_list)))
        diag_dict = dict(zip(diag_list, diag_num))

        return diag_dict

    except FileNotFoundError:
        print("File " + DIAG_FILENAME + " has not found.\nThe specified folder (" + DATA_PATH +
              ") must contain files with data (" + DATA_FILENAME +
              ") and file with structure of diagnosis (" + DIAG_FILENAME + ").")
        sys.exit(0)


def load_dataset(folder_path=DATA_PATH):
    if not os.path.exists(folder_path + PKL_FILENAME):
        xy = parser(folder_path)
        fix_bw(xy, folder_path)

    with open(folder_path + PKL_FILENAME, 'rb') as infile:
        dataset = pkl.load(infile)

    return dataset


def load_holter(patient=0, folder_path=HOLTER_PATH):
    with open(folder_path + HOLTER_FILENAME + str(patient) + ".pkl", 'rb') as infile:
        dataset = pkl.load(infile)
    return dataset


def load_good_holter():
    x1 = load_holter(0)[1884000:1910500]

    x2 = load_holter(0)[368000: 549500]
    #x2 = load_holter(0)[1590000:1617000]
    return x1, x2

def load_mit():
    with open(MIT_PATH, 'rb') as infile:
        dataset = pkl.load(infile)[:,:,:1]
    return dataset



def fix_bw(xy, folder_path):
    print("Baseline wondering fixing is started. It's take some time.")

    X = xy["x"]
    patients_num = X.shape[0]
    for i in range(patients_num):
        print("\rSignal %s/" % str(i + 1) + str(patients_num) + ' is fixed.', end='')
        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x'] = X

    with open(folder_path + PKL_FILENAME, 'rb') as outfile:
        pkl.dump(xy, outfile)

    print("The dataset is saved.")


def normalize_data(X):
    mn = X.mean(axis=0)
    st = X.std(axis=0)
    x_std = np.zeros(X.shape)
    for i in range(X.shape[0]):
        x_std[i] = (X[i] - mn) / st
    return x_std


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]
    diag_dict = get_diag_dict()

    X = X / 1000.0

    print(X.shape)
    print(Y.shape)

    for i in np.arange(X.shape[0] - 1, 0, - 1):
        diags_norm_rythm = [0]
        diags_fibrilation = [15]
        diags_flutter = [16, 17, 18]
        diags_hypertrophy = [119, 120]
        diags_extrasystole = [69, 70, 71, 72, 73, 74, 75, 86]

        title = ""
        if Y[i, 0] == 1: title += "norm rythm"
        if Y[i, 15] == 1: title += "\nfibrilation"
        for j in diags_flutter:
            if Y[i, j] == 1:
                title += "\nflutter"
                break
        for j in diags_hypertrophy:
            if Y[i, j] == 1:
                title += "\nhypertrophy"
                break
        for j in diags_extrasystole:
            if Y[i, j] == 1:
                title += "\nextrasystole"
                break

        plt.figure(i, figsize=[12, 8])
        plt.suptitle(title)

        plt.subplot(6, 2, 1)
        plt.plot(X[i, :, 0], color='black')
        plt.gca().set_title(LEADS_NAMES[0])

        plt.subplot(6, 2, 3)
        plt.plot(X[i, :, 1], color='black')
        plt.gca().set_title(LEADS_NAMES[1])

        plt.subplot(6, 2, 5)
        plt.plot(X[i, :, 2], color='black')
        plt.gca().set_title(LEADS_NAMES[2])

        plt.subplot(6, 2, 7)
        plt.plot(X[i, :, 3], color='black')
        plt.gca().set_title(LEADS_NAMES[3])

        plt.subplot(6, 2, 9)
        plt.plot(X[i, :, 4], color='black')
        plt.gca().set_title(LEADS_NAMES[4])

        plt.subplot(6, 2, 11)
        plt.plot(X[i, :, 5], color='black')
        plt.gca().set_title(LEADS_NAMES[5])

        plt.subplot(6, 2, 2)
        plt.plot(X[i, :, 6], color='black')
        plt.gca().set_title(LEADS_NAMES[6])

        plt.subplot(6, 2, 4)
        plt.plot(X[i, :, 7], color='black')
        plt.gca().set_title(LEADS_NAMES[7])

        plt.subplot(6, 2, 6)
        plt.plot(X[i, :, 8], color='black')
        plt.gca().set_title(LEADS_NAMES[8])

        plt.subplot(6, 2, 8)
        plt.plot(X[i, :, 9], color='black')
        plt.gca().set_title(LEADS_NAMES[9])

        plt.subplot(6, 2, 10)
        plt.plot(X[i, :, 10], color='black')
        plt.gca().set_title(LEADS_NAMES[10])

        plt.subplot(6, 2, 12)
        plt.plot(X[i, :, 11], color='black')
        plt.gca().set_title(LEADS_NAMES[11])

        plt.show()
