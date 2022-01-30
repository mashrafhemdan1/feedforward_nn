import os, sys, tarfile, shutil, cv2
import cupy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

test_size = 100

def extract_images(base_url='/content/drive/My Drive/PDML Assignments/A1'):
    tar_url = base_url + '/flower_photos.tgz'
    tar = tarfile.open(tar_url, 'r')
    extract_path= base_url
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

def sort_images(dataset_filename = '/content/drive/My Drive/PDML Assignments/A1/flower_photos'):
    flowers = os.listdir(dataset_filename)
    for flower in flowers:
        if os.path.isdir(os.path.join(dataset_filename, flower)):
            print("Flower: ", flower)
            flower_dir = os.path.join(dataset_filename, flower)
            files = os.listdir(flower_dir)
            sorted_files = sorted(files)

            # Training part
            train_dir = os.path.join(flower_dir, 'training')
            print("Training dir: ", train_dir)
            if not os.path.isdir(train_dir):
                os.mkdir(train_dir)
            for i, file_name in enumerate(sorted_files[:-test_size]):
                full_file_name = os.path.join(flower_dir, file_name)
                print("filename (", i, "): ", full_file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, train_dir)
                
            # Testing part
            test_dir = os.path.join(flower_dir, 'testing')
            print("Testing dir: ", test_dir)
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            for i, file_name in enumerate(sorted_files[-test_size:]):
                full_file_name = os.path.join(flower_dir, file_name)
                print("filename (", i, "): ", full_file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, test_dir)

def read_images(base_dir = '/content/drive/My Drive/PDML Assignments/A1/flower_photos'):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for label in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, label)):
            continue
        train_path = base_dir+'/'+label+'/training'
        test_path = base_dir+'/'+label+'/testing'

        for imgfile in os.listdir(train_path):
            img_arr = cv2.imread(os.path.join(train_path, imgfile))
            X_train.append(img_arr)
            y_train.append(label)

        for imgfile in os.listdir(test_path):
            img_arr = cv2.imread(os.path.join(test_path, imgfile))
            X_test.append(img_arr)
            y_test.append(label)
        
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test

def resize_images(X_train, X_test, dim = (128, 128)):
    X_train_t = np.empty((X_train.shape[0], dim[0], dim[1], 3))
    for i, x in enumerate(X_train):
        X_train_t[i] = np.array(cv2.resize(np.asnumpy(x), dim))
    del X_train
    X_test_t = np.empty((X_test.shape[0], dim[0], dim[1], 3))
    for i, x in enumerate(X_test):
        X_test_t[i] = np.array(cv2.resize(np.asnumpy(x), dim))
    del X_test
    X_train = np.array(X_train_t)
    X_test = np.array(X_test_t)

    return X_train, X_test

def get_minsize(X_train, X_test):
    min_width = 1000
    min_height = 1000
    for x in X_train:
        if x.shape[0] < min_width:
            min_width = x.shape[0]
        if x.shape[1] < min_height:
            min_height = x.shape[1]

    for x in X_test:
        if x.shape[0] < min_width:
            min_width = x.shape[0]
        if x.shape[1] < min_height:
            min_height = x.shape[1]
    
    return min_width, min_height

def flatten_images(X_train, X_test):
    X_train_t = X_train.reshape((X_train.shape[0], -1))
    X_test_t = X_test.reshape((X_test.shape[0], -1))
    return X_train_t, X_test_t

def shuffle_images(X_train, X_test, y_train, y_test):
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    return X_train, X_test, y_train, y_test

def save_as_np(X_train, X_test, y_train, y_test, base_dir = '/content/drive/My Drive/PDML Assignments/A1/flower_photos'):
    np.save(os.path.join(base_dir, 'X_train'), X_train)
    np.save(os.path.join(base_dir, 'y_train'), y_train)
    np.save(os.path.join(base_dir, 'X_test'), X_test)
    np.save(os.path.join(base_dir, 'y_test'), y_test)

def load_np(base_dir = '/content/drive/My Drive/PDML Assignments/A1/flower_photos'):
    X_train = np.load(os.path.join(base_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(base_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def plot_graphs(training_loss, training_accuracy, validation_loss, validation_accuracy, epochs):
    epochs_range = np.asnumpy(np.arange(0, epochs, 1))
    plt.plot(epochs_range, np.asnumpy(training_loss), label = "Training Loss", color='blue')
    plt.plot(epochs_range, np.asnumpy(validation_loss), label = "Validation Loss", color='red')
    plt.legend()
    plt.show()
    plt.plot(epochs_range, np.asnumpy(training_accuracy), label = "Training Accuracy", color='blue')
    plt.plot(epochs_range, np.asnumpy(validation_accuracy), label = "Validation Accuracy", color='red')
    plt.legend()
    plt.show()