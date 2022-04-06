from cProfile import label
from random import random
from matplotlib.pyplot import cla
import numpy
from src.classification.features import *
from src.classification.constants import *
from src.classification.classificator import *
from src.classification.voting import *


# Fitting parameters
def fit(train, test, method, SHOW=False):
    if method not in [histogram, dft, dct, gradient, scale]:
        return []
    param = (0, 0, 0)
    if method == histogram:
        param = (8, 30, 3)
    if method == dft or method == dct:
        param = (6, 30, 3)
    if method == gradient:
        param = (2, 30, 3)
    if method == scale:
        param = (0.05, 0.5, 0.05)

    classificator = Classificator()

    best_param = param[0]
    classf = classificator.test_classify(train, test, method, best_param)
    stat = [[best_param], [classf]]

    for i in np.arange(param[0] + param[2], param[1], param[2]):
        new_classf = classificator.test_classify(
            train, test, method, i)
        stat[0].append(i)
        stat[1].append(new_classf)
        if new_classf > classf:
            classf = new_classf
            best_param = i

    if SHOW:
        plt.plot(stat[0], stat[1], label=method.__name__)
    return [best_param, classf], stat


def cross_validation(data, etalons_range=[5, 6], SHOW=False):
    res = []
    start, end = etalons_range
    faces_in_train_stats = {}
    vote_stats = []

    for size in range(start, end + 1):
        print(f"{size} FACES IN TRAIN, {10 - size} FACES IN TEST")
        X_train, X_test, y_train, y_test = DataPreprocessor.split_data_randomly(
            data, size)
        train = DataPreprocessor.mesh_data([X_train, y_train])
        test = DataPreprocessor.mesh_data([X_test, y_test])
        parameters = {}

        for method in ALL_METHODS:
            print("-" * 50)
            print(f"fitting params for {method}...")
            results = fit(
                train, test, METHOD_TO_FUNCTION_MAP[method], SHOW=SHOW)
            if method in faces_in_train_stats:
                faces_in_train_stats[method].append(results[0][1])
            else:
                faces_in_train_stats[method] = [results[0][1]]
            parameters[METHOD_TO_PARAM_MAP[method]] = results[0][0]
            print(
                f"for {method} got param {results[0][0]} with score {results[0][1]}")
        if SHOW:
            plt.title(f"train size={size}"), plt.legend(
                loc='best'), plt.xlabel("parameter"), plt.ylabel("score")
            plt.show()
            # draw_splited_data(X_train, X_test, size)
        print("-" * 50)
        print(f"Result of fitting for all methods: {parameters}")
        print("Voting...")
        classf = test_voting(train, test, parameters)
        vote_stats.append(classf)
        print(f"voted accuracy: {classf}")
        res.append([parameters, classf])

    if SHOW:
        for method, stats in faces_in_train_stats.items():
            plt.plot(range(1, len(stats) + 1), stats, label=method)
            plt.legend(loc='best'), plt.xlabel(
                "train size"), plt.ylabel("score")
        plt.show()
        plt.plot(range(1, len(vote_stats) + 1),
                 vote_stats),
        plt.title("Voting")
        plt.xlabel("train size"), plt.ylabel("voted_score")
        plt.show()

    best_res = [[], 0]
    best = 0
    for i in range(end - start + 1):
        if res[i][1] > best:
            best = res[i][1]
            best_res[0] = res[i][0]
            best_res[1] = i + start
    best_res.append(best)

    print('BEST_RES', best_res)
    return best_res


def mean_cross_validation(data):
    method_results = {}

    for method in ALL_METHODS:
        print(f"method {method}...")
        all_results = []
        for i in range(0, 10):
            print("-" * 50)
            print(f"iteration {i}...")
            X_train, X_test, y_train, y_test = DataPreprocessor.split_data_randomly(
                data, 1)
            train = DataPreprocessor.mesh_data([X_train, y_train])
            test = DataPreprocessor.mesh_data([X_test, y_test])
            print("-" * 50)
            print(f"fitting params for {method}...")
            results = fit(train, test, METHOD_TO_FUNCTION_MAP[method])
            all_results.append(results[0][1])
        method_results[method] = all_results

    for k, v in method_results.items():
        xs = range(0, len(v))
        ys = v
        r = random()
        b = random()
        g = random()
        color = (r, g, b)
        plt.plot(xs, ys, color=color, label=k)
    plt.title('Method Deviation')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.show()

    for k, v in method_results.items():
        xs = range(0, len(v))
        ys = [numpy.mean(v)] * len(v)
        r = random()
        b = random()
        g = random()
        color = (r, g, b)
        plt.plot(xs, ys, color=color, label=k)
    plt.title('Method Mean')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Score')
    plt.legend(loc='best')
    plt.show()


def mean_voting(data, etalons_range=[5, 6], SHOW=False):
    res = []

    for _ in range(0, 10):
        start, end = etalons_range
        faces_in_train_stats = {}

        for size in range(start, end + 1):
            print(f"{size} FACES IN TRAIN, {10 - size} FACES IN TEST")
            X_train, X_test, y_train, y_test = DataPreprocessor.split_data_randomly(
                data, size)
            train = DataPreprocessor.mesh_data([X_train, y_train])
            test = DataPreprocessor.mesh_data([X_test, y_test])
            parameters = {}

            for method in ALL_METHODS:
                print("-" * 50)
                print(f"fitting params for {method}...")
                results = fit(
                    train, test, METHOD_TO_FUNCTION_MAP[method], SHOW=False)
                if method in faces_in_train_stats:
                    faces_in_train_stats[method].append(results[0][1])
                else:
                    faces_in_train_stats[method] = [results[0][1]]
                parameters[METHOD_TO_PARAM_MAP[method]] = results[0][0]
                print(
                    f"for {method} got param {results[0][0]} with score {results[0][1]}")
            print("Voting...")
            classf = test_voting(train, test, parameters)
        res.append(classf)

    print(res, [np.mean(res)] * len(res))

    if SHOW:
        plt.plot(range(0, len(res)), res),
        plt.plot(range(0, len(res)), [np.mean(res)] * len(res))
        plt.title("Voting")
        plt.xlabel("iteration"), plt.ylabel("voted_score")
        plt.show()
