from __future__ import annotations

from collections import OrderedDict

from PySide6 import QtCore, QtGui, QtWidgets


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isalnum())


def _action_text(action: QtGui.QAction | None) -> str:
    if action is None:
        return ""
    return str(action.text()).replace("&", "")


class RibbonCommandButton(QtWidgets.QToolButton):
    """Ribbon-aware tool button with stable role metadata for tests and future tabs."""

    def __init__(
        self,
        action: QtGui.QAction,
        *,
        role: str,
        text_override: str | None = None,
        object_name: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("ribbonCommandRole", str(role))
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setDefaultAction(action)
        if text_override:
            self.setText(str(text_override))
        if object_name:
            self.setObjectName(str(object_name))

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        if str(role) == "primary":
            self.setMinimumSize(148, 56)
        else:
            self.setMinimumSize(128, 24)


class RibbonGroup(QtWidgets.QFrame):
    """Ribbon group with separate primary and compact command lanes."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = str(title)
        self.setObjectName(f"ribbonGroup{_normalize_name(self._title)}")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 4)
        layout.setSpacing(6)

        self._body = QtWidgets.QWidget(self)
        self._body.setObjectName(f"{self.objectName()}Body")
        self._body_layout = QtWidgets.QHBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(6)
        layout.addWidget(self._body)

        title_label = QtWidgets.QLabel(self._title, self)
        title_label.setObjectName(f"{self.objectName()}Title")
        title_font = title_label.font()
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(title_label)

        self._primary_buttons: list[RibbonCommandButton] = []
        self._compact_buttons: list[RibbonCommandButton] = []
        self._compact_columns: list[QtWidgets.QWidget] = []
        self._current_compact_column_layout: QtWidgets.QVBoxLayout | None = None
        self._compact_count_in_column = 0

    def title(self) -> str:
        return self._title

    def add_primary_action(
        self,
        action: QtGui.QAction,
        *,
        text_override: str | None = None,
        object_name: str | None = None,
    ) -> RibbonCommandButton:
        button = RibbonCommandButton(
            action,
            role="primary",
            text_override=text_override,
            object_name=object_name,
            parent=self,
        )
        self._body_layout.addWidget(button, 0, QtCore.Qt.AlignTop)
        self._primary_buttons.append(button)
        return button

    def add_compact_action(
        self,
        action: QtGui.QAction,
        *,
        text_override: str | None = None,
        object_name: str | None = None,
    ) -> RibbonCommandButton:
        if self._current_compact_column_layout is None or self._compact_count_in_column >= 3:
            column_widget = QtWidgets.QWidget(self._body)
            column_widget.setObjectName(f"{self.objectName()}CompactColumn{len(self._compact_columns) + 1}")
            column_layout = QtWidgets.QVBoxLayout(column_widget)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(4)
            self._body_layout.addWidget(column_widget, 0, QtCore.Qt.AlignTop)
            self._compact_columns.append(column_widget)
            self._current_compact_column_layout = column_layout
            self._compact_count_in_column = 0

        button = RibbonCommandButton(
            action,
            role="compact",
            text_override=text_override,
            object_name=object_name,
            parent=self,
        )
        self._current_compact_column_layout.addWidget(button)
        self._compact_buttons.append(button)
        self._compact_count_in_column += 1
        return button

    def primary_action_texts(self) -> list[str]:
        return [_action_text(button.defaultAction()) for button in self._primary_buttons]

    def compact_action_texts(self) -> list[str]:
        return [_action_text(button.defaultAction()) for button in self._compact_buttons]


class RibbonPage(QtWidgets.QWidget):
    """A first-class ribbon page that owns titled command groups."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = str(title)
        self.setObjectName(f"ribbonPage{_normalize_name(self._title)}")

        self._groups: OrderedDict[str, RibbonGroup] = OrderedDict()
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 8)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)

    def title(self) -> str:
        return self._title

    def add_group(self, group: RibbonGroup) -> RibbonGroup:
        insert_at = max(0, self._layout.count() - 1)
        if self._groups:
            separator = QtWidgets.QFrame(self)
            separator.setObjectName(f"{group.objectName()}Separator")
            separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            self._layout.insertWidget(insert_at, separator, 0, QtCore.Qt.AlignVCenter)
            insert_at += 1
        self._layout.insertWidget(insert_at, group, 0, QtCore.Qt.AlignTop)
        self._groups[group.title()] = group
        return group

    def group_titles(self) -> list[str]:
        return list(self._groups.keys())

    def group(self, title: str) -> RibbonGroup | None:
        return self._groups.get(str(title))


class CollapsibleRibbonHost(QtWidgets.QWidget):
    """Tabbed, collapsible ribbon host with first-class pages and groups."""

    collapseToggleRequested = QtCore.Signal(bool)
    collapsedChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mainRibbonHost")
        self._collapsed = False
        self._pages: OrderedDict[str, RibbonPage] = OrderedDict()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QWidget(self)
        header.setObjectName("ribbonHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)

        self._tab_bar = QtWidgets.QTabBar(header)
        self._tab_bar.setObjectName("ribbonTabBar")
        self._tab_bar.setDrawBase(False)
        self._tab_bar.setDocumentMode(True)
        self._tab_bar.currentChanged.connect(self._on_current_tab_changed)
        header_layout.addWidget(self._tab_bar, 1)

        self._collapse_toggle_button = QtWidgets.QToolButton(header)
        self._collapse_toggle_button.setObjectName("ribbonCollapseToggleButton")
        self._collapse_toggle_button.clicked.connect(self._request_toggle_collapsed)
        header_layout.addWidget(self._collapse_toggle_button, 0)

        layout.addWidget(header)

        self._content_stack = QtWidgets.QStackedWidget(self)
        self._content_stack.setObjectName("ribbonContentStack")
        layout.addWidget(self._content_stack)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._sync_collapse_ui()

    def add_page(self, page: RibbonPage) -> None:
        title = page.title()
        self._pages[title] = page
        self._content_stack.addWidget(page)
        self._tab_bar.addTab(title)
        if self._tab_bar.count() == 1:
            self._tab_bar.setCurrentIndex(0)
            self._content_stack.setCurrentWidget(page)

    def add_tab(self, title: str, content: QtWidgets.QWidget) -> None:
        if isinstance(content, RibbonPage):
            page = content
        else:
            page = RibbonPage(title, self)
            wrapper_group = RibbonGroup(str(title), page)
            placeholder = QtWidgets.QWidget(page)
            placeholder_layout = QtWidgets.QVBoxLayout(placeholder)
            placeholder_layout.setContentsMargins(0, 0, 0, 0)
            placeholder_layout.addWidget(content)
            wrapper_group._body_layout.addWidget(placeholder)
            page.add_group(wrapper_group)
        self.add_page(page)

    def tab_titles(self) -> list[str]:
        return [str(self._tab_bar.tabText(index)) for index in range(self._tab_bar.count())]

    def page_titles(self) -> list[str]:
        return list(self._pages.keys())

    def current_tab_title(self) -> str:
        current_index = self._tab_bar.currentIndex()
        if current_index < 0:
            return ""
        return str(self._tab_bar.tabText(current_index))

    def page(self, title: str) -> RibbonPage | None:
        return self._pages.get(str(title))

    def current_page(self) -> RibbonPage | None:
        return self.page(self.current_tab_title())

    def is_collapsed(self) -> bool:
        return bool(self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed_value = bool(collapsed)
        if self._collapsed == collapsed_value:
            return
        self._collapsed = collapsed_value
        self._sync_collapse_ui()
        self.collapsedChanged.emit(self._collapsed)

    def _request_toggle_collapsed(self) -> None:
        self.collapseToggleRequested.emit(not self._collapsed)

    def _on_current_tab_changed(self, index: int) -> None:
        if index >= 0:
            self._content_stack.setCurrentIndex(index)

    def _sync_collapse_ui(self) -> None:
        self._content_stack.setVisible(not self._collapsed)
        self._collapse_toggle_button.setText("Expand Ribbon" if self._collapsed else "Collapse Ribbon")
        self.updateGeometry()
