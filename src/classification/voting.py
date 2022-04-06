from src.classification.constants import P_IMAGES_PER_PERSON, PARAMS
from src.classification.features import *
from src.classification.constants import *
from src.classification.classificator import Classificator
from src.utils.data_preprocessor import DataPreprocessor


def voting(train, test, parameters, SHOW=False):
    result = {}
    voted_answers = []
    classifier = Classificator()

    for method in ALL_METHODS:
        result[method] = classifier.classify(
            train, test, METHOD_TO_FUNCTION_MAP[method], parameters[METHOD_TO_PARAM_MAP[method]])

    for i in range(len(test[0])):
        answers_to_image_1 = {}

        for method in result:
            answer = result[method][i]
            if answer in answers_to_image_1:
                answers_to_image_1[answer] += 1
            else:
                answers_to_image_1[answer] = 1

        best_size = sorted(answers_to_image_1.items(),
                           key=lambda item: item[1], reverse=True)[0]
        voted_answers.append(best_size[0])

        if SHOW:
            for image, person in zip(train[0], train[1]):
                if person == best_size[0]:
                    plt.subplot(1, 2, 1)
                    plt.imshow(test[0][i], cmap="gray"), plt.xticks(
                        []), plt.yticks([]), plt.title('Query Image')
                    plt.subplot(1, 2, 2)
                    plt.imshow(image, cmap="gray"), plt.xticks(
                        []), plt.yticks([]), plt.title('Result')
                    plt.suptitle(f"Voting")
                    plt.show()
                    break

    return voted_answers


def vote_classifier(data, parameters, SHOW=False):
    # parameters, train_size = cross_validation(data)
    train_size = parameters[P_IMAGES_PER_PERSON]

    x_train, x_test, y_train, y_test = DataPreprocessor.split_data(
        data, train_size, DRAW=SHOW)
    train = DataPreprocessor.mesh_data([x_train, y_train])
    test = DataPreprocessor.mesh_data([x_test, y_test])
    return voting(train, test, parameters, SHOW=SHOW)


def test_voting(train, test, parameters):
    res = voting(train, test, parameters)
    sum = 0
    for i in range(len(test[0])):
        if test[1][i] == res[i]:
            sum += 1
    return sum / len(test[0])
