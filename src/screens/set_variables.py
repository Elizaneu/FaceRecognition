from pickletools import read_uint1
import tkinter as tki
from ..ui.window import Window
from ..ui.ui import UI
from ..classification.constants import *

INPUTS = [
    (P_HISTOGRAM),
    (P_DCT),
    (P_DFT),
    (P_GRADIENT),
    (P_SCALE),
    (P_IMAGES_PER_PERSON)
]

B_APPLY = 'B_APPLY'
B_RESET = 'B_RESET'


class SetVariablesScreen(Window):
    def __init__(self) -> None:
        super().__init__("AI Analysis", width=400, height=350)
        self.__init_components()

    def __get_parameters_values(self) -> None:

        for input in INPUTS:
            name = input
            value = float(self.get_component(name).get() or 0)

            # TODO: store castings in constants
            PARAMS[name] = int(value) if name != P_SCALE else float(value)

        self.close()

    def __reset_parameters_values(self) -> None:
        for input in INPUTS:
            name = input

            entry = self.get_component(name)
            entry.delete(0, tki.END)
            entry.insert(0, PARAMS_DEFAULTS[name])

    def __init_components(self) -> None:
        for i in range(0, len(INPUTS)):
            name = INPUTS[i]

            row = i

            self.include_component(
                name,
                UI.get_label(name, row, 0),
            )
            entry = self.include_component(
                name,
                UI.get_entry(row, 1),
            )
            entry.insert(0, PARAMS[name])

        self.include_component(
            B_APPLY,
            UI.get_button(
                "Apply", self.__get_parameters_values, len(INPUTS) + 1, 0),
        )
        self.include_component(
            B_RESET,
            UI.get_button(
                "Restore", self.__reset_parameters_values, len(INPUTS) + 1, 1),
        )
