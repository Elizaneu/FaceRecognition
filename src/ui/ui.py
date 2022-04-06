from cProfile import label
import tkinter as tki
from tkinter.font import BOLD


GRID_GAP = (8, 8)


class UI:
    def get_entry(row=0, column=0) -> tki.Entry:
        def make(window):
            entry = tki.Entry(window, width=20, bg='white',
                              foreground='black', selectforeground='black')
            entry.grid(row=row, column=column,
                       padx=GRID_GAP[1], pady=8)

            return entry

        return make

    def get_label(title: str, row=0, column=0) -> tki.Label:
        def make(window):
            label = tki.Label(window, text=title)
            label.grid(row=row, column=column,
                       padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return label

        return make

    def get_title(title: str, row=0, column=0) -> tki.Label:
        def make(window):
            label = tki.Label(window, text=title,
                              font=("Helvetica", 24, BOLD))
            label.grid(row=row, column=column,
                       padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return label

        return make

    def get_button(text: str, on_click, row=0, column=0, width=10) -> tki.Button:
        def make(window):
            button = tki.Button(window, text=text,
                                width=width, command=on_click, highlightbackground='#3E4149')

            button.grid(row=row, column=column,
                        padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return button

        return make

    def get_radio(text: str, value: str, on_click, variable=None, row=0, column=0, width=10) -> tki.Radiobutton:
        def make(window):
            radio = tki.Radiobutton(window, value=value, variable=variable, text=text, width=width,
                                    command=on_click)

            radio.grid(row=row, column=column,
                       padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return radio

        return make


UI.get_entry = staticmethod(UI.get_entry)
UI.get_label = staticmethod(UI.get_label)
UI.get_button = staticmethod(UI.get_button)
UI.get_radio = staticmethod(UI.get_radio)
