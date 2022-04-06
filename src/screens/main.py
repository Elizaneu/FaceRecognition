from pickle import TRUE
from tkinter import StringVar, Tk, filedialog
from src.classification.voting import voting
from src.classification.cross_validation import cross_validation, mean_cross_validation, mean_voting
from src.classification.constants import *
from src.screens.set_variables import SetVariablesScreen
from src.utils.data_preprocessor import DataPreprocessor
from src.ui.window import Window
from src.ui.ui import UI
from src.classification.classificator import Classificator
from src.classification.features import *
from src.screens.warning import Warning
from cv2 import *

# Images uploading
T_IMAGES_UPLOADING = "T_IMAGES_UPLOADING"
L_IMAGES_UPLOADED_HEAD = "LABEL_IMAGES_UPLOADED_HEAD"
L_IMAGES_UPLOADED_VALUE = "LABEL_IMAGES_UPLOADED_VALUE"
B_UPLOAD_IMAGES = "B_UPLOAD_IMAGES"

# Parameters settings
T_PARAMETERS_SETTINGS = "T_PARAMETERS_SETTINGS"
B_SET_PARAMETERS = "B_SET_PARAMETERS"

# Method selection
T_METHOD = "T_METHOD"

# Classification
T_CLASSIFICATION = "T_CLASSIFICATION"
B_CLASSIFY = "B_CLASSIFY"
B_CLASSIFY_IMAGE = "B_CLASSIFY_IMAGE"
B_PRINT_METHODS = "B_PRINT_METHODS"
B_CROSS_VALIDATION = "B_CROSS_VALIDATION"
B_MEAN_CROSS_VALIDATION = "B_MEAN_CROSS_VALIDATION"
B_MEAN_VOTING = "B_MEAN_VOTING"
B_VOTING = "B_VOTING"
B_TEST_PARALLEL_SYSTEM = "B_TEST_PARALLEL_SYSTEM"
L_TEST_CLASSIFICATION_SCORES = "L_TEST_CLASSIFICATION_SCORES"
L_ALL_RESULTS = "L_CLASSIFICATION_SCORES"

T_RESULTS = "T_RESULTS"
B_CLEAR_RESULTS = "B_CLEAR_RESULTS"
L_SCORE_ = "L_SCORE_"


class MainScreen(Window):
    def __init__(self) -> None:
        super().__init__("Face Recognition")

        self.__selected_method = StringVar(value=R_METHOD_HISTOGRAM)
        self.__selected_image = []

        self.__init_components()

        # Recognition
        self.__is_data_loaded = False
        self.__data = []
        self.__classification_results = {}
        self.__classification_index = 0

    # UI
    def __init_components(self):
        # Images uploading
        self.include_component(
            T_IMAGES_UPLOADING,
            UI.get_title("Images uploading", self._get_next_row())
        )
        self.include_component(
            L_IMAGES_UPLOADED_HEAD,
            UI.get_label("Images status:", self._get_next_row()),
        )
        self.include_component(
            L_IMAGES_UPLOADED_VALUE,
            UI.get_label("not loaded", self._get_row(), 1),
        )
        self.include_component(
            B_UPLOAD_IMAGES,
            UI.get_button(
                "Load images", self.__handle_images_load, self._get_next_row()
            ),
        )
        # Preferences setting
        self.include_component(
            T_PARAMETERS_SETTINGS,
            UI.get_title("Parameters settings", self._get_next_row()),
        )
        self.include_component(
            B_SET_PARAMETERS,
            UI.get_button("Set parameters values",
                          self.__handle_parameters_set, self._get_next_row(), 0, 15)
        )
        # Method selection
        self.include_component(
            T_METHOD,
            UI.get_title("Method selection", self._get_next_row()),
        )
        self._get_next_row()
        self.include_component(
            R_METHOD_HISTOGRAM,
            UI.get_radio(text="Histogram", value=R_METHOD_HISTOGRAM, variable=self.__selected_method, on_click=self.__handle_method_select,
                         row=self._get_row())
        )
        self.include_component(
            R_METHOD_DCT,
            UI.get_radio(text="DCT", value=R_METHOD_DCT, variable=self.__selected_method, on_click=self.__handle_method_select,
                         row=self._get_row(), column=1)
        )
        self.include_component(
            R_METHOD_DFT,
            UI.get_radio(text="DFT", value=R_METHOD_DFT, variable=self.__selected_method, on_click=self.__handle_method_select,
                         row=self._get_row(), column=2)
        )
        self.include_component(
            R_METHOD_GRADIENT,
            UI.get_radio(text="Gradient", value=R_METHOD_GRADIENT, variable=self.__selected_method, on_click=self.__handle_method_select,
                         row=self._get_row(), column=3)
        )
        self.include_component(
            R_METHOD_SCALE,
            UI.get_radio(text="Scale", value=R_METHOD_SCALE, variable=self.__selected_method, on_click=self.__handle_method_select,
                         row=self._get_row(), column=4)
        )
        # Classification
        self.include_component(
            T_CLASSIFICATION,
            UI.get_title("Image classification", self._get_next_row()),
        )
        self.include_component(
            B_CLASSIFY,
            UI.get_button("Test Classify", self.__handle_test_classify,
                          self._get_next_row())
        )
        self.include_component(
            B_CLASSIFY_IMAGE,
            UI.get_button("Classify Image", self.__handle_classify,
                          self._get_row(), column=1)
        )
        self.include_component(
            B_PRINT_METHODS,
            UI.get_button("Print Image", self.__handle_print_methods_to_image,
                          self._get_row(), column=2)
        )
        self.include_component(
            B_CROSS_VALIDATION,
            UI.get_button("Cross Validate", self.__handle_cross_validation,
                          self._get_row(), column=3)
        )
        self.include_component(
            B_VOTING,
            UI.get_button("Voting",
                          self.__handle_voting, self._get_row(), column=4)
        )
        self.include_component(
            B_MEAN_CROSS_VALIDATION,
            UI.get_button("Mean Cross Validate",
                          self.__handle_mean_cross_validation, self._get_row(), column=5)
        )
        self.include_component(
            B_MEAN_VOTING,
            UI.get_button("Mean Voting",
                          self.__handle_mean_voting, self._get_row(), column=6)
        )
        self._get_next_row()
        self.include_component(
            B_TEST_PARALLEL_SYSTEM,
            UI.get_button("Test Parallel System",
                          self.__handle_test_parallel_system, self._get_row(), column=0)
        )
        self.include_component(
            T_RESULTS,
            UI.get_title("Results", self._get_next_row())
        )
        self.include_component(
            B_CLEAR_RESULTS,
            UI.get_button("Clear results",
                          self.__handle_clear_results, self._get_row(), 1)
        )
        self._get_next_row()

    def __add_score(self, score) -> None:
        scores_per_row = 5
        row = self._get_row() + self.__classification_index // scores_per_row

        self.include_component(
            f"{L_SCORE_}{self.__classification_index}",
            UI.get_label(f"#{self.__classification_index + 1}: {score}",
                         row, self.__classification_index % scores_per_row)
        )

        self.__classification_index += 1

    def __handle_images_load(self) -> None:
        self.__data = DataPreprocessor.load_images('./faces/s')
        self.__is_data_loaded = True

        label = self.get_component(L_IMAGES_UPLOADED_VALUE)
        label.configure(text="loaded")

    def __handle_method_select(self) -> None:
        print(self.__selected_method.get())

    def __handle_parameters_set(self) -> None:
        modal = SetVariablesScreen()
        modal.open()

    def __handle_test_classify(self) -> None:
        current_method_name = self.__selected_method.get()
        current_method = METHOD_TO_FUNCTION_MAP[current_method_name]
        current_method_params = METHOD_TO_PARAM_MAP[current_method_name]

        if not self.__is_data_loaded:
            modal = Warning("You need to load images before classification!")
            modal.open()
            return

        x_train, x_test, y_train, y_test = DataPreprocessor.split_data(
            self.__data, PARAMS[P_IMAGES_PER_PERSON]
        )
        train = DataPreprocessor.mesh_data([x_train, y_train])
        test = DataPreprocessor.mesh_data([x_test, y_test])

        classificator = Classificator()
        score = classificator.test_classify(train, test, current_method,
                                            PARAMS[current_method_params])

        self.__add_score(score)

    def __handle_classify(self) -> None:
        if not self.__is_data_loaded:
            modal = Warning("You need to load images before classification!")
            modal.open()
            return

        x_train, _, y_train, _ = DataPreprocessor.split_data(
            self.__data, PARAMS[P_IMAGES_PER_PERSON]
        )
        train = DataPreprocessor.mesh_data([x_train, y_train])
        self.update()
        image_path = filedialog.askopenfilename(initialdir="./",
                                                title="Select a File",
                                                filetypes=[("Image files", "*.jpg *.png *.pgm")])
        self.update()
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        image = image / 255
        self.__selected_image = image

        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap="gray"), plt.xticks(
            []), plt.yticks([]), plt.title(f'Query Image')

        index = 2

        classificator = Classificator()

        for method in ALL_METHODS:
            current_method = METHOD_TO_FUNCTION_MAP[method]
            current_method_params = METHOD_TO_PARAM_MAP[method]
            score = classificator.classify_with_single_answer(train, ([image], [0]), current_method,
                                                              PARAMS[current_method_params])
            self.__classification_results[method] = score
            plt.subplot(2, 3, index)
            index += 1
            plt.imshow(score, cmap="gray"), plt.xticks([]), plt.yticks(
                []), plt.title(f"{method}, {PARAMS[current_method_params]}")

        plt.suptitle(f"train size={PARAMS[P_IMAGES_PER_PERSON]}")
        plt.show()
        self.update()

    def __handle_print_methods_to_image(self) -> None:
        if not len(self.__selected_image):
            modal = Warning(
                "You need to select an image before printing methods to image!")
            modal.open()
            return

        plt.subplot(2, 3, 1)
        plt.imshow(self.__selected_image, cmap="gray"), plt.title("Original")

        for index, method in zip(range(2, 7), ALL_METHODS):
            plt.subplot(2, 3, index)
            param = PARAMS[METHOD_TO_PARAM_MAP[method]]

            if method == R_METHOD_HISTOGRAM:
                plt.hist(self.__selected_image.flatten(),
                         bins=param, range=(0, 1))
            elif method == R_METHOD_GRADIENT:
                res = SHOW_METHOD_TO_PARAM_MAP[method](
                    self.__selected_image, param)
                plt.plot(range(0, len(res)), res)
            else:
                plt.imshow(SHOW_METHOD_TO_PARAM_MAP[method](
                    self.__selected_image, param), cmap="gray")

            plt.title(f'{method}, {param}')

        plt.show()

    def __handle_clear_results(self) -> None:
        if self.__classification_index == 0:
            return

        for i in range(self.__classification_index):
            self.remove_component(f"{L_SCORE_}{i}")

        self.__classification_index = 0

    def __handle_cross_validation(self) -> None:
        parameters, best_size, score = cross_validation(
            self.__data, etalons_range=[1, 4], SHOW=True)
        text = f"best size: {best_size}, best_params: {parameters}, score: {score}"
        modal = Warning(text)
        modal.open()

    def __handle_mean_cross_validation(self) -> None:
        mean_cross_validation(self.__data)

    def __handle_voting(self) -> None:
        train_size = PARAMS[P_IMAGES_PER_PERSON]

        x_train, x_test, y_train, y_test = DataPreprocessor.split_data(
            self.__data, train_size, DRAW=True)
        train = DataPreprocessor.mesh_data([x_train, y_train])
        test = DataPreprocessor.mesh_data([x_test, y_test])
        voting(train, test, PARAMS, SHOW=True)

    def __handle_mean_voting(self) -> None:
        mean_voting(self.__data, [1, 1], True)

    def __handle_test_parallel_system(self) -> None:
        x_train, x_test, y_train, y_test = DataPreprocessor.split_data(
            self.__data, PARAMS[P_IMAGES_PER_PERSON], DRAW=False)
        train = [x_train, y_train]
        result = []
        count = 0
        sum = 0

        for test_image, true_answer in zip(x_test, y_test):
            res = voting(train, [[test_image], [true_answer]], PARAMS)
            if true_answer == res[0]:
                sum += 1
            else:
                print(f"!!!!!!!")
                plt.subplot(2, 3, 1)
                plt.imshow(test_image, cmap="gray"), plt.xticks(
                    []), plt.yticks([]), plt.title(f'Query Image')
                index = 2
                classificator = Classificator()

                for method in ALL_METHODS:
                    current_method = METHOD_TO_FUNCTION_MAP[method]
                    current_method_params = METHOD_TO_PARAM_MAP[method]
                    score = classificator.classify_with_single_answer(train, ([test_image], [0]), current_method,
                                                                      PARAMS[current_method_params])
                    self.__classification_results[method] = score
                    plt.subplot(2, 3, index)
                    index += 1
                    plt.imshow(score, cmap="gray"), plt.xticks([]), plt.yticks(
                        []), plt.title(f"{method}, {PARAMS[current_method_params]}")
                plt.suptitle(f"train size={PARAMS[P_IMAGES_PER_PERSON]}")
                plt.show()
            count += 1
            result.append(sum / count)
            print(f"{count} images --> {sum / count}")
        plt.plot(range(1, len(result) + 1),  result)
        plt.xlabel("amount of test images")
        plt.ylabel("score")
        plt.title("Voting")
        plt.show()
