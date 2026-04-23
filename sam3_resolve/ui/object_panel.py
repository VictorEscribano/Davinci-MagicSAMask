"""
Object panel — right panel upper section.

Each tracked object gets an ObjectRow: color swatch, number badge,
editable name, eye/trash icons, opacity slider, feather slider.
The active row has a 2px left accent border.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sam3_resolve.constants import MAX_OBJECTS
from sam3_resolve.ui.canvas_widget import OBJECT_COLORS

logger = logging.getLogger(__name__)


# ── Single object row ──────────────────────────────────────────────────────

class ObjectRow(QWidget):
    """
    UI row for one tracked object.

    Signals:
        selected(int):              User clicked the row → make active
        visibility_toggled(int, bool)
        deleted(int)
        color_changed(int, QColor)
        opacity_changed(int, int)   object_id, 0-100
        feather_changed(int, int)   object_id, 0-20
        name_changed(int, str)
    """

    selected            = pyqtSignal(int)
    visibility_toggled  = pyqtSignal(int, bool)
    deleted             = pyqtSignal(int)
    color_changed       = pyqtSignal(int, QColor)
    opacity_changed     = pyqtSignal(int, int)
    feather_changed     = pyqtSignal(int, int)
    name_changed        = pyqtSignal(int, str)

    def __init__(
        self,
        object_id: int,
        color: QColor,
        name: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.object_id = object_id
        self._color = color
        self._visible = True
        self._active = False

        self.setObjectName("object_row")
        self.setFixedHeight(88)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build(name or f"Object {object_id}")

    def _build(self, name: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # ── Top row: swatch + number + name + eye + trash ─────
        top = QHBoxLayout()
        top.setSpacing(6)

        # Colour swatch
        self._swatch = QPushButton()
        self._swatch.setFixedSize(16, 16)
        self._swatch.setToolTip("Click to change colour")
        self._apply_swatch_color(self._color)
        self._swatch.clicked.connect(self._pick_color)

        # Number badge
        badge = QLabel(str(self.object_id))
        badge.setFixedSize(18, 18)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "background-color: #3A3A3A; border-radius: 2px; "
            "color: #E8E8E8; font-size: 10px; font-weight: 700;"
        )

        # Name (inline edit on pencil click)
        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet("color: #E8E8E8; font-size: 12px;")
        self._name_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._name_edit = QLineEdit(name)
        self._name_edit.setFixedHeight(18)
        self._name_edit.hide()
        self._name_edit.returnPressed.connect(self._commit_name)
        self._name_edit.editingFinished.connect(self._commit_name)

        pencil = QPushButton("✎")
        pencil.setObjectName("toolbar_btn")
        pencil.setFixedSize(18, 18)
        pencil.setToolTip("Rename")
        pencil.clicked.connect(self._start_edit)

        # Eye toggle
        self._eye_btn = QPushButton("👁")
        self._eye_btn.setObjectName("toolbar_btn")
        self._eye_btn.setFixedSize(22, 22)
        self._eye_btn.setToolTip("Toggle visibility")
        self._eye_btn.setCheckable(True)
        self._eye_btn.setChecked(True)
        self._eye_btn.toggled.connect(self._on_visibility)

        # Delete
        trash = QPushButton("✕")
        trash.setObjectName("toolbar_btn")
        trash.setFixedSize(22, 22)
        trash.setToolTip("Delete object")
        trash.setStyleSheet("color: #E85050;")
        trash.clicked.connect(lambda: self.deleted.emit(self.object_id))

        top.addWidget(self._swatch)
        top.addWidget(badge)
        top.addWidget(self._name_lbl)
        top.addWidget(self._name_edit)
        top.addWidget(pencil)
        top.addWidget(self._eye_btn)
        top.addWidget(trash)
        outer.addLayout(top)

        # ── Opacity slider ────────────────────────────────────
        op_row = QHBoxLayout()
        op_row.setSpacing(6)
        op_lbl = QLabel("Opacity")
        op_lbl.setObjectName("info_key")
        op_lbl.setFixedWidth(44)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(70)
        self._opacity_val = QLabel("70%")
        self._opacity_val.setObjectName("info_value")
        self._opacity_val.setFixedWidth(32)
        self._opacity_slider.valueChanged.connect(self._on_opacity)
        op_row.addWidget(op_lbl)
        op_row.addWidget(self._opacity_slider)
        op_row.addWidget(self._opacity_val)
        outer.addLayout(op_row)

        # ── Feather slider ────────────────────────────────────
        fe_row = QHBoxLayout()
        fe_row.setSpacing(6)
        fe_lbl = QLabel("Feather")
        fe_lbl.setObjectName("info_key")
        fe_lbl.setFixedWidth(44)
        self._feather_slider = QSlider(Qt.Orientation.Horizontal)
        self._feather_slider.setRange(0, 20)
        self._feather_slider.setValue(0)
        self._feather_val = QLabel("0 px")
        self._feather_val.setObjectName("info_value")
        self._feather_val.setFixedWidth(32)
        self._feather_slider.valueChanged.connect(self._on_feather)
        fe_row.addWidget(fe_lbl)
        fe_row.addWidget(self._feather_slider)
        fe_row.addWidget(self._feather_val)
        outer.addLayout(fe_row)

    # ── Active state ──────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        self._active = active
        border = "2px solid #4A9EFF" if active else "none"
        bg = "#1A2A3A" if active else "transparent"
        self.setStyleSheet(
            f"QWidget#object_row {{ border-left: {border}; background-color: {bg}; }}"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.selected.emit(self.object_id)
        super().mousePressEvent(event)

    # ── Slot handlers ─────────────────────────────────────────────────────

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Choose object colour")
        if color.isValid():
            self._color = color
            self._apply_swatch_color(color)
            self.color_changed.emit(self.object_id, color)

    def _apply_swatch_color(self, color: QColor) -> None:
        self._swatch.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #555555; "
            "border-radius: 2px;"
        )

    def _on_visibility(self, checked: bool) -> None:
        self._visible = checked
        self._eye_btn.setStyleSheet("" if checked else "color: #555555;")
        self.visibility_toggled.emit(self.object_id, checked)

    def _on_opacity(self, val: int) -> None:
        self._opacity_val.setText(f"{val}%")
        self.opacity_changed.emit(self.object_id, val)

    def _on_feather(self, val: int) -> None:
        self._feather_val.setText(f"{val} px")
        self.feather_changed.emit(self.object_id, val)

    def _start_edit(self) -> None:
        self._name_lbl.hide()
        self._name_edit.setText(self._name_lbl.text())
        self._name_edit.show()
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _commit_name(self) -> None:
        text = self._name_edit.text().strip() or f"Object {self.object_id}"
        self._name_lbl.setText(text)
        self._name_edit.hide()
        self._name_lbl.show()
        self.name_changed.emit(self.object_id, text)

    # ── Public getters ────────────────────────────────────────────────────

    @property
    def opacity(self) -> int:
        return self._opacity_slider.value()

    @property
    def feather(self) -> int:
        return self._feather_slider.value()

    @property
    def name(self) -> str:
        return self._name_lbl.text()

    @property
    def color(self) -> QColor:
        return self._color


# ── Object panel container ─────────────────────────────────────────────────

class ObjectPanel(QWidget):
    """
    Full right-panel objects section.
    Manages a list of ObjectRow widgets, enforces MAX_OBJECTS limit,
    and emits aggregated signals to the canvas and main window.
    """

    active_object_changed   = pyqtSignal(int)
    object_added            = pyqtSignal(int)
    object_deleted          = pyqtSignal(int)
    visibility_changed      = pyqtSignal(int, bool)
    opacity_changed         = pyqtSignal(int, int)
    feather_changed         = pyqtSignal(int, int)
    color_changed           = pyqtSignal(int, QColor)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: dict[int, ObjectRow] = {}
        self._active_id: int = -1
        self._next_id: int = 1
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(10, 8, 10, 4)
        self._title = QLabel("OBJECTS (0 / 8)")
        self._title.setObjectName("section_label")
        self._add_btn = QPushButton("+ Add Object")
        self._add_btn.setFixedHeight(22)
        self._add_btn.clicked.connect(self.add_object)
        header.addWidget(self._title)
        header.addStretch()
        header.addWidget(self._add_btn)
        layout.addLayout(header)

        # Rows container
        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        self._rows_layout.addStretch()
        layout.addWidget(self._rows_widget)

    # ── Public API ─────────────────────────────────────────────────────────

    def add_object(self, color: Optional[QColor] = None) -> int:
        """
        Add a new object row. Returns the new object ID, or -1 if at MAX.
        """
        if len(self._rows) >= MAX_OBJECTS:
            return -1

        obj_id = self._next_id
        self._next_id += 1
        c = color or OBJECT_COLORS[(obj_id - 1) % len(OBJECT_COLORS)]

        row = ObjectRow(obj_id, c, parent=self)
        row.selected.connect(self._on_selected)
        row.visibility_toggled.connect(self.visibility_changed)
        row.deleted.connect(self.delete_object)
        row.opacity_changed.connect(self.opacity_changed)
        row.feather_changed.connect(self.feather_changed)
        row.color_changed.connect(self.color_changed)

        # Insert before stretch
        self._rows_layout.insertWidget(
            self._rows_layout.count() - 1, row
        )
        self._rows[obj_id] = row
        self._update_title()
        self._on_selected(obj_id)
        self.object_added.emit(obj_id)
        return obj_id

    def delete_object(self, object_id: int) -> None:
        if object_id not in self._rows:
            return
        row = self._rows.pop(object_id)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._update_title()
        self.object_deleted.emit(object_id)
        if self._active_id == object_id and self._rows:
            self._on_selected(next(iter(self._rows)))

    def set_active_object(self, object_id: int) -> None:
        self._on_selected(object_id)

    def object_count(self) -> int:
        return len(self._rows)

    def get_row(self, object_id: int) -> Optional[ObjectRow]:
        return self._rows.get(object_id)

    # ── Internals ─────────────────────────────────────────────────────────

    def _on_selected(self, object_id: int) -> None:
        for oid, row in self._rows.items():
            row.set_active(oid == object_id)
        self._active_id = object_id
        self.active_object_changed.emit(object_id)

    def _update_title(self) -> None:
        n = len(self._rows)
        self._title.setText(f"OBJECTS ({n} / {MAX_OBJECTS})")
        self._add_btn.setEnabled(n < MAX_OBJECTS)
