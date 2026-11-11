"""A small preview of the source file, so you can go back to what was analysed.

Images get a thumbnail (click to enlarge). Video gets a thumbnail frame and, on Play, a small
player. Audio gets a compact player. Findings with time regions can start playback at that moment.
If Qt's multimedia support is missing, the widget degrades to thumbnails and says so.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QRectF, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..core.finding import MediaType
from ..core.frames import thumbnail
from .spectro_view import fmt_time

_MEDIA = None


def media_modules():
    """The Qt multimedia classes, imported only when first needed (loading native multimedia
    libraries can crash on machines without audio or graphics hardware, so nothing touches them
    until a playable file is opened). False when unavailable."""
    global _MEDIA
    if os.environ.get("MITRAAV_NO_MEDIA"):     # headless machines (CI): the audio stack can crash
        return False
    if _MEDIA is None:
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
            _MEDIA = dict(Player=QMediaPlayer, Audio=QAudioOutput, Devices=QMediaDevices, Video=QVideoWidget)
        except Exception:                      # pragma: no cover - depends on the Qt install
            _MEDIA = False
    return _MEDIA


def media_available() -> bool:
    return bool(media_modules())


THUMB_W, THUMB_H = 280, 150
AUDIO_H = 48


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGB")
    qi = QImage(img.tobytes(), img.width, img.height, 3 * img.width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


def load_thumbnail(path: Path, box: tuple[int, int] = (THUMB_W * 2, THUMB_H * 2)) -> QPixmap | None:
    """Thumbnail of an image file (EXIF orientation applied); None if it cannot be read."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail(box)
            return pil_to_pixmap(im)
    except Exception:
        return None


class _ThumbWorker(QThread):
    """Extracts one video frame off the GUI thread (large files can take a moment to seek)."""
    done = Signal(str, QImage)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self) -> None:
        f = thumbnail(self.path, width=THUMB_W * 2)
        if f and f.exists():
            self.done.emit(str(self.path), QImage(str(f)))


class _BoxedImage(QLabel):
    """A picture that rescales with its window and can outline regions (boxes in fractions)."""

    def __init__(self, pix: QPixmap | None, boxes: list[dict] | None = None):
        super().__init__()
        self.pix = pix
        self.boxes = boxes or []
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if pix is None:
            self.setText("Cannot show this image")

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._redraw()

    def _redraw(self) -> None:
        if self.pix is None:
            return
        pm = self.pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self.boxes:
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor("#fbbf24"))
            pen.setWidth(2)
            for b in self.boxes:
                x0, y0 = b["x0"] * pm.width(), b["y0"] * pm.height()
                w, h = (b["x1"] - b["x0"]) * pm.width(), (b["y1"] - b["y0"]) * pm.height()
                p.setPen(pen)
                p.drawRect(QRectF(x0 - 3, y0 - 3, w + 6, h + 6))
            p.end()
        self.setPixmap(pm)

    def box_at(self, x: float, y: float) -> dict | None:
        """The box under a point in this widget's coordinates, for tooltips."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        ox, oy = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        fx, fy = (x - ox) / pm.width(), (y - oy) / pm.height()
        return next((b for b in self.boxes if b["x0"] <= fx <= b["x1"] and b["y0"] <= fy <= b["y1"]), None)

    def mouseMoveEvent(self, e) -> None:
        b = self.box_at(e.position().x(), e.position().y())
        self.setToolTip(b["label"] if b else "")
        super().mouseMoveEvent(e)


class _FullImage(QDialog):
    def __init__(self, path: Path, parent=None, boxes: list[dict] | None = None):
        super().__init__(parent)
        self.setWindowTitle(path.name + (f" - {len(boxes)} text region(s) outlined" if boxes else ""))
        pix = load_thumbnail(path, (3000, 2400))
        self.view = _BoxedImage(pix, boxes)
        self.view.setMouseTracking(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        self.resize(min(1100, (pix.width() if pix else 400) + 40), min(800, (pix.height() if pix else 200) + 40))


class MediaPreview(QFrame):
    """Thumbnail or player for the file being analysed."""
    frame_requested = Signal(float)          # seconds into the video (0 for a picture)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self.path: Path | None = None
        self.media: MediaType | None = None
        self.player = None
        self.audio_out = None
        self.video = None                              # created on the first Play of a video
        self._workers: list[_ThumbWorker] = []
        self._dragging = False

        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # follows the pane, never forces it
        self._thumb_src: QPixmap | None = None
        self.thumb.setCursor(Qt.PointingHandCursor)
        self.thumb.mousePressEvent = lambda e: self._thumb_clicked()
        self.audio_page = QLabel("♪")
        self.audio_page.setAlignment(Qt.AlignCenter)
        self.audio_page.setObjectName("audioglyph")
        self.audio_page.setStyleSheet("font-size: 34px; color: #8e9096;")
        self.stack = QStackedWidget()
        for w in (self.thumb, self.audio_page):
            self.stack.addWidget(w)

        self.play_btn = QPushButton()
        self.play_btn.setObjectName("secondary")
        self.play_btn.setFixedWidth(40)
        self._play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self._pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self.play_btn.setIcon(self._play_icon)
        self.play_btn.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.time = QLabel("0:00 / 0:00")
        self.time.setObjectName("hint")
        self.note = QLabel("")
        self.note.setObjectName("hint")
        self.note.setWordWrap(True)
        self.controls = QWidget()
        cl = QHBoxLayout(self.controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.play_btn)
        cl.addWidget(self.slider, 1)
        cl.addWidget(self.time)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        lay.addWidget(self.stack)
        lay.addWidget(self.controls)
        self.frame_btn = QPushButton("Scan this frame")
        self.frame_btn.setObjectName("secondary")
        self.frame_btn.setToolTip("Run the picture probes on the frame you are looking at, or on a region of it")
        self.frame_btn.clicked.connect(lambda: self.frame_requested.emit(self.position_seconds()))
        lay.addWidget(self.frame_btn)
        lay.addWidget(self.note)

        self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.slider.sliderReleased.connect(self._released)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._drain_workers)   # never leave a thread running at exit
        self.clear()

    # --- lazy native pieces ---
    def _ensure_player(self) -> bool:
        if self.player:
            return True
        m = media_modules()
        if not m:
            return False
        self.player = m["Player"](self)
        if m["Devices"].audioOutputs():                # no sound card: play silently instead of crashing
            self.audio_out = m["Audio"](self)
            self.audio_out.setVolume(0.8)
            self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._position)
        self.player.durationChanged.connect(self._duration)
        self.player.playbackStateChanged.connect(self._state)
        self.player.errorOccurred.connect(self._error)
        return True

    def _ensure_video(self) -> None:
        if self.video is None and self._ensure_player():
            self.video = media_modules()["Video"]()
            self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # its size hint is huge
            self.stack.addWidget(self.video)
            self.player.setVideoOutput(self.video)

    def _drain_workers(self) -> None:
        for w in list(self._workers):
            w.wait(3000)

    # --- loading ---
    def clear(self) -> None:
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())
        self.path, self.media = None, None
        self.controls.setVisible(False)
        self.frame_btn.setVisible(False)
        self.note.setText("")
        self.setVisible(False)

    def load(self, path: Path, media: MediaType | None) -> None:
        self.clear()
        self.path, self.media = Path(path), media
        self.slider.setRange(0, 0)
        self.time.setText("0:00 / 0:00")
        self.play_btn.setIcon(self._play_icon)
        if media == MediaType.IMAGE:
            pix = load_thumbnail(self.path)
            if pix:
                self._set_thumb(pix)
                self.thumb.setToolTip("Click to enlarge")
            else:
                self.thumb.setText("Cannot show this image")
            self.stack.setCurrentWidget(self.thumb)
            self._sizing(resizable=True)
            self.controls.setVisible(False)
        elif media == MediaType.VIDEO:
            self._thumb_src = None
            self.thumb.clear()
            self.thumb.setText("video")
            self.thumb.setToolTip("Click to play")
            self.stack.setCurrentWidget(self.thumb)
            self._sizing(resizable=True)
            worker = _ThumbWorker(self.path)
            worker.done.connect(self._thumb_ready)
            worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
            self._workers.append(worker)               # held until it finishes, so it is never destroyed running
            worker.start()
            self._open_player()
        elif media == MediaType.AUDIO:
            self.stack.setCurrentWidget(self.audio_page)
            self._sizing(resizable=False)
            self._open_player()
        else:
            return
        self.frame_btn.setVisible(media in (MediaType.IMAGE, MediaType.VIDEO))
        self.frame_btn.setText("Scan this frame" if media == MediaType.VIDEO else "Scan a region")
        self.setVisible(True)

    def _open_player(self) -> None:
        if not self._ensure_player():
            self.controls.setVisible(False)
            self.note.setText("Playback is unavailable (Qt multimedia was not found).")
            return
        self.controls.setVisible(True)
        if self.audio_out is None:
            self.note.setText("No audio output device found: playback will be silent.")
        self.player.setSource(QUrl.fromLocalFile(str(self.path)))

    def _thumb_ready(self, path: str, img: QImage) -> None:
        if self.path and str(self.path) == path and self.stack.currentWidget() is self.thumb:
            self._set_thumb(QPixmap.fromImage(img))

    def _sizing(self, resizable: bool) -> None:
        """Images and video grow with the pane (drag the splitter or maximise); audio stays a slim bar."""
        if resizable:
            self.stack.setMinimumHeight(90)
            self.stack.setMaximumHeight(16777215)
            self.setMaximumHeight(16777215)
        else:
            self.stack.setFixedHeight(AUDIO_H)
            self.setMaximumHeight(AUDIO_H + 90)

    def _set_thumb(self, pix: QPixmap) -> None:
        self._thumb_src = pix
        self._fit_thumb()

    def _fit_thumb(self) -> None:
        if self._thumb_src is not None and not self._thumb_src.isNull() and self.thumb.width() > 8:
            self.thumb.setPixmap(self._thumb_src.scaled(self.thumb.size(), Qt.KeepAspectRatio,
                                                        Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit_thumb()

    def _thumb_clicked(self) -> None:
        if self.media == MediaType.IMAGE and self.path:
            _FullImage(self.path, self.window()).exec()
        elif self.media == MediaType.VIDEO:
            self.toggle()

    # --- playback ---
    def toggle(self) -> None:
        if not self.player or not self.path:
            return
        if self.player.playbackState() == self.player.PlaybackState.PlayingState:
            self.player.pause()
            return
        if self.media == MediaType.VIDEO:
            self._ensure_video()
            if self.video is not None:
                self.stack.setCurrentWidget(self.video)
        self.player.play()

    def play_from(self, seconds: float) -> bool:
        """Start playing at `seconds` (file time). False when there is nothing to play."""
        if not self.player or not self.path or self.media not in (MediaType.AUDIO, MediaType.VIDEO):
            return False
        if self.media == MediaType.VIDEO:
            self._ensure_video()
            if self.video is not None:
                self.stack.setCurrentWidget(self.video)
        self.player.setPosition(max(0, int(seconds * 1000)))
        self.player.play()
        return True

    def position_seconds(self) -> float:
        return self.player.position() / 1000.0 if self.player and self.media == MediaType.VIDEO else 0.0

    def can_play(self) -> bool:
        return bool(self.player and self.path and self.media in (MediaType.AUDIO, MediaType.VIDEO))

    def _position(self, ms: int) -> None:
        if not self._dragging:
            self.slider.setValue(ms)
        self.time.setText(f"{fmt_time(ms / 1000)} / {fmt_time(self.player.duration() / 1000)}")

    def _duration(self, ms: int) -> None:
        self.slider.setRange(0, max(0, ms))

    def _released(self) -> None:
        self._dragging = False
        self.player.setPosition(self.slider.value())

    def _state(self, state) -> None:
        playing = state == self.player.PlaybackState.PlayingState
        self.play_btn.setIcon(self._pause_icon if playing else self._play_icon)

    def _error(self, error, message: str = "") -> None:
        self.note.setText("This file cannot be played here (" + (message or "unsupported format") + ").")
