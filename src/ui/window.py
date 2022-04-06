import tkinter as tki
from typing import Any
from src.ui.ui import UI

# Window
WIDTH = 1240
HEIGHT = 1240

# Key codes
KEY_ESCAPE = '<Escape>'

# Events
EV_WINDOW_CLOSE = 'WM_DELETE_WINDOW'


class Window:
    def __init__(self, title, width=WIDTH, height=HEIGHT) -> None:
        # Base window configuration
        self.__tki = tki.Tk()
        self.__tki.title(title)
        self.__tki.geometry(f"{width}x{height}")

        # Default event handlers
        self.__tki.bind(KEY_ESCAPE, lambda: self.__handle_close)
        self.__tki.protocol(EV_WINDOW_CLOSE, self.__handle_close)

        # Components
        self.__components = {}
        self.__ui_row = 0

    def open(self) -> None:
        self.__tki.update_idletasks()
        self.__tki.mainloop()

    def close(self) -> None:
        self.__handle_close()

    def update(self) -> None:
        self.__tki.update()

    def include_component(self, name, make_component) -> Any:
        component = make_component(self.__tki)

        self.__components[f"{name}"] = component

        return component

    def remove_component(self, name) -> None:
        component = self.get_component(name)
        component.destroy()

        del self.__components[f"{name}"]

    def get_component(self, name) -> Any:
        return self.__components[f"{name}"]

    def add_gutter(self) -> None:
        self.include_component(
            "GUTTER",
            UI.get_label("", self._get_next_row())
        )

    def _get_row(self) -> int:
        return self.__ui_row

    def _get_next_row(self) -> int:
        self.__ui_row = self.__ui_row + 1

        return self.__ui_row

    def __handle_close(self) -> None:
        self.__tki.destroy()
