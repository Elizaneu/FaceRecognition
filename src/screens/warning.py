from src.ui.window import Window
from src.ui.ui import UI


T_TITLE = "T_TITLE"
B_OK = "B_OK"


class Warning(Window):
    def __init__(self, warning: str) -> None:
        super().__init__("Warning!", width=len(warning) * 12, height=100)
        self.__init_components(warning)

    def __init_components(self, warning: str) -> None:
        self.include_component(
            T_TITLE,
            UI.get_title(warning, self._get_next_row())
        )
        self.include_component(
            B_OK,
            UI.get_button("OK", lambda: self.close(),
                          self._get_next_row(), width=int(len(warning) * 1.2))
        )
