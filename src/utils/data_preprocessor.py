import random as rnd
import cv2
import matplotlib.pyplot as plt


class DataPreprocessor:
    def load_images(path, ext=".pgm"):
        dt_images = []
        dt_targets = []

        for i in range(1, 41):
            for j in range(1, 11):
                image = cv2.imread(f"{path}{i}/{j}{ext}")
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dt_images.append(image / 255)
                    dt_targets.append(i - 1)

        return [dt_images, dt_targets]

    def mesh_data(data):
        indexes = rnd.sample(range(0, len(data[0])), len(data[0]))

        return [data[0][index] for index in indexes], [data[1][index] for index in indexes]

    def draw_splited_data(x_train, x_test, images_per_person_in_train, rows=5):
        i, j = 0, 0
        plt.rcParams.update({'font.size': 7})
        list_of_classes = [rows]

        for rows in list_of_classes:
            draw_train = images_per_person_in_train
            index = 1
            draw_test = 10 - images_per_person_in_train

            while index != rows * 10 + 1:
                if draw_train > 0:
                    plt.subplot(rows, 10, index)
                    plt.imshow(x_train[i], cmap='gray'), plt.xticks(
                        []), plt.yticks([])
                    if index <= 10:
                        plt.title('Train')
                    draw_train -= 1
                    i += 1
                    index += 1
                elif draw_test > 0:
                    plt.subplot(rows, 10, index)
                    plt.imshow(x_test[j], cmap='gray'), plt.xticks(
                        []), plt.yticks([])
                    if index <= 10:
                        plt.title('Test')
                    draw_test -= 1
                    j += 1
                    index += 1
                else:
                    draw_train = images_per_person_in_train
                    draw_test = 10 - images_per_person_in_train
            plt.show()

    def split_data(data, images_per_person_in_train=5, DRAW=False):
        images_per_person = 10
        images_all = len(data[0])

        x_train, x_test, y_train, y_test = [], [], [], []

        for i in range(0, images_all, images_per_person):
            x_train.extend(data[0][i: i + images_per_person_in_train])
            y_train.extend(data[1][i: i + images_per_person_in_train])

            x_test.extend(
                data[0][i + images_per_person_in_train: i + images_per_person])
            y_test.extend(
                data[1][i + images_per_person_in_train: i + images_per_person])

        if DRAW:
            DataPreprocessor.draw_splited_data(
                x_train, x_test, images_per_person_in_train)

        return x_train, x_test, y_train, y_test

    def split_data_randomly(data, images_per_person_in_train=5, DRAW=False):
        images_per_person = 10
        amount_of_images = len(data[0])

        x_train, x_test, y_train, y_test = [], [], [], []

        for i in range(0, amount_of_images, images_per_person):
            indexes = list(range(i, i + images_per_person))
            train_indexes = rnd.sample(indexes, images_per_person_in_train)
            x_train.extend([data[0][index] for index in train_indexes])
            y_train.extend([data[1][index] for index in train_indexes])

            test_indexes = set(indexes) - set(train_indexes)
            x_test.extend([data[0][index] for index in test_indexes])
            y_test.extend([data[1][index] for index in test_indexes])

        if DRAW:
            DataPreprocessor.draw_splited_data(
                x_train, x_test, images_per_person_in_train)

        return x_train, x_test, y_train, y_test


DataPreprocessor.load_images = staticmethod(
    DataPreprocessor.load_images)
DataPreprocessor.mesh_data = staticmethod(DataPreprocessor.mesh_data)
DataPreprocessor.draw_splited_data = staticmethod(
    DataPreprocessor.draw_splited_data)
DataPreprocessor.split_data = staticmethod(DataPreprocessor.split_data)
DataPreprocessor.split_data_randomly = staticmethod(
    DataPreprocessor.split_data_randomly)
