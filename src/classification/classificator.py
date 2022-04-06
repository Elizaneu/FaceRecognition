from src.classification.features import *


class Classificator:
    def __create_feature(self, images, method, parameter):
        return [method(image, parameter) for image in images]

    def __distance(self, element1, element2):
        return np.linalg.norm(np.array(element1) - np.array(element2))

    def classify_with_single_answer(self, train, test, method, parameter):
        if method not in [histogram, dft, dct, gradient, scale]:
            return []

        featured_train = self.__create_feature(train[0], method, parameter)
        featured_test = self.__create_feature(test[0], method, parameter)
        answer = []

        for test_element in featured_test:
            min_element = [100000, -1]

            for i in range(len(featured_train)):
                dist = self.__distance(test_element, featured_train[i])
                if dist < min_element[0]:
                    min_element = [dist, i]

            if min_element[1] < 0:
                answer = train[0][0]
            else:
                answer = train[0][min_element[1]]

        return answer

    def classify(self, train, test, method, parameter):
        if method not in [histogram, dft, dct, gradient, scale]:
            return []
        featured_train = self.__create_feature(train[0], method, parameter)
        featured_test = self.__create_feature(test[0], method, parameter)
        answers = []

        for test_element in featured_test:
            min_element = [100000, -1]

            for i in range(len(featured_train)):
                dist = self.__distance(test_element, featured_train[i])
                if dist < min_element[0]:
                    min_element = [dist, i]

            if min_element[1] < 0:
                answers.append(0)
            else:
                answers.append(train[1][min_element[1]])

        return answers

    def test_classify(self, train, test, method, parameter):
        if method not in [histogram, dft, dct, gradient, scale]:
            return []
        answers = self.classify(train, test, method, parameter)
        correct_answers = 0

        for i in range(len(answers)):
            if answers[i] == test[1][i]:
                correct_answers += 1

        return correct_answers / len(answers)
