# desktop_app.py  (PySide6, Python 3.13 friendly, Chinese, modern chat UI)
import os, sys, asyncio, traceback
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from agent_core import PlaywrightMCPAgent, Config, Logger, LogLevel


# ----------------------------
# UI Design Tokens (High contrast dark)
# ----------------------------
class Palette:
    BG = "#0A1220"
    PANEL = "#101C2E"
    PANEL_2 = "#0F2138"
    BORDER = "rgba(255,255,255,0.12)"
    BORDER_2 = "rgba(255,255,255,0.18)"

    TEXT = "rgba(255,255,255,0.95)"
    MUTED = "rgba(255,255,255,0.72)"
    MUTED_2 = "rgba(255,255,255,0.58)"

    ACCENT = "#79A8FF"
    ACCENT_2 = "#4C8DFF"

    GOOD = "#35C66B"
    BAD = "#FF5C5C"
    WARN = "#FFB020"

    USER_BUBBLE = "#234A86"              # 用户气泡：更亮的蓝
    BOT_BUBBLE = "#182C46"               # 助手气泡：蓝灰
    LOG_BUBBLE = "rgba(255,255,255,0.08)"


class MsgRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    LOG = "log"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    role: MsgRole
    text: str
    ts: str
    pending: bool = False


def now_hhmm() -> str:
    return QtCore.QTime.currentTime().toString("HH:mm")


# ----------------------------
# Chat Model
# ----------------------------
class ChatModel(QtCore.QAbstractListModel):
    Role = QtCore.Qt.UserRole + 1
    Text = QtCore.Qt.UserRole + 2
    Time = QtCore.Qt.UserRole + 3
    Pending = QtCore.Qt.UserRole + 4

    def __init__(self):
        super().__init__()
        self.items: List[ChatMessage] = []

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.items)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        m = self.items[index.row()]
        if role == QtCore.Qt.DisplayRole:
            return m.text
        if role == self.Role:
            return m.role.value
        if role == self.Text:
            return m.text
        if role == self.Time:
            return m.ts
        if role == self.Pending:
            return m.pending
        return None

    def add(self, msg: ChatMessage):
        self.beginInsertRows(QtCore.QModelIndex(), len(self.items), len(self.items))
        self.items.append(msg)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.items.clear()
        self.endResetModel()

    def update_last_pending(self, new_text: str, pending: bool):
        # 找到最后一个 pending assistant 气泡替换
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i].role == MsgRole.ASSISTANT and self.items[i].pending:
                self.items[i].text = new_text
                self.items[i].pending = pending
                idx = self.index(i)
                self.dataChanged.emit(idx, idx, [self.Text, self.Pending, QtCore.Qt.DisplayRole])
                return


# ----------------------------
# Bubble Delegate (Paint chat bubbles)
# ----------------------------
class BubbleDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_width_ratio = 0.74

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        role = index.data(ChatModel.Role)
        text = index.data(ChatModel.Text) or ""
        ts = index.data(ChatModel.Time) or ""
        pending = bool(index.data(ChatModel.Pending))

        rect = option.rect.adjusted(12, 8, -12, -8)
        view = option.widget
        max_w = int(view.viewport().width() * self.max_width_ratio)

        # Typography
        font = option.font
        font.setPointSize(10)
        painter.setFont(font)
        fm = QtGui.QFontMetrics(font)

        pad_x, pad_y = 14, 10
        meta_h = 16

        text_flags = QtCore.Qt.TextWordWrap | QtCore.Qt.TextWrapAnywhere
        text_rect = fm.boundingRect(QtCore.QRect(0, 0, max_w - pad_x * 2, 10_000), text_flags, text)

        bubble_w = min(max_w, text_rect.width() + pad_x * 2)
        bubble_h = text_rect.height() + pad_y * 2 + meta_h

        # Colors per role
        if role == MsgRole.USER.value:
            bubble_x = rect.right() - bubble_w
            bubble_color = QtGui.QColor(Palette.USER_BUBBLE)
            meta_color = QtGui.QColor(255, 255, 255, 170)
            text_color = QtGui.QColor(255, 255, 255, 242)
            stroke = QtGui.QColor(255, 255, 255, 28)
        elif role == MsgRole.ASSISTANT.value:
            bubble_x = rect.left()
            bubble_color = QtGui.QColor(Palette.BOT_BUBBLE)
            meta_color = QtGui.QColor(255, 255, 255, 170)
            text_color = QtGui.QColor(255, 255, 255, 242)
            stroke = QtGui.QColor(255, 255, 255, 28)
        elif role == MsgRole.LOG.value:
            bubble_x = rect.left()
            bubble_color = QtGui.QColor(Palette.LOG_BUBBLE)
            meta_color = QtGui.QColor(255, 255, 255, 140)
            text_color = QtGui.QColor(255, 255, 255, 200)
            stroke = QtGui.QColor(255, 255, 255, 20)
        else:  # SYSTEM
            bubble_x = rect.left()
            bubble_color = QtGui.QColor(121, 168, 255, 26)  # accent tint
            meta_color = QtGui.QColor(255, 255, 255, 150)
            text_color = QtGui.QColor(255, 255, 255, 235)
            stroke = QtGui.QColor(121, 168, 255, 60)

        bubble = QtCore.QRect(bubble_x, rect.top(), bubble_w, bubble_h)

        # Shadow
        shadow_color = QtGui.QColor(0, 0, 0, 90)
        shadow = QtCore.QRect(bubble.x() + 1, bubble.y() + 2, bubble.width(), bubble.height())
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(shadow_color)
        painter.drawRoundedRect(shadow, 14, 14)

        # Bubble
        painter.setBrush(bubble_color)
        painter.setPen(QtGui.QPen(stroke, 1))
        painter.drawRoundedRect(bubble, 14, 14)

        # Text
        text_area = bubble.adjusted(pad_x, pad_y, -pad_x, -(pad_y + meta_h))
        painter.setPen(text_color)
        painter.drawText(text_area, text_flags, text)

        # Meta
        meta_text = ts + (" · 正在思考…" if pending else "")
        meta_area = bubble.adjusted(pad_x, bubble.height() - meta_h - 6, -pad_x, -6)
        meta_font = option.font
        meta_font.setPointSize(8)
        painter.setFont(meta_font)
        painter.setPen(meta_color)
        painter.drawText(meta_area, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, meta_text)

        painter.restore()

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        text = index.data(ChatModel.Text) or ""

        view = option.widget
        max_w = int(view.viewport().width() * self.max_width_ratio)

        font = option.font
        font.setPointSize(10)
        fm = QtGui.QFontMetrics(font)
        pad_x, pad_y = 14, 10
        meta_h = 16

        text_flags = QtCore.Qt.TextWordWrap | QtCore.Qt.TextWrapAnywhere
        text_rect = fm.boundingRect(QtCore.QRect(0, 0, max_w - pad_x * 2, 10_000), text_flags, text)

        bubble_h = text_rect.height() + pad_y * 2 + meta_h
        return QtCore.QSize(option.rect.width(), bubble_h + 16)


# ----------------------------
# Worker Thread (async agent)
# ----------------------------
class AgentWorker(QtCore.QThread):
    log = QtCore.Signal(str)
    reply = QtCore.Signal(str)
    status = QtCore.Signal(str)
    connected = QtCore.Signal(bool)
    error = QtCore.Signal(str)

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.loop = None
        self.agent: Optional[PlaywrightMCPAgent] = None
        self.q = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._amain())

    async def _amain(self):
        self.q = asyncio.Queue()
        try:
            self.agent = PlaywrightMCPAgent(self.api_key)
            self.agent.logger = Logger("Agent", LogLevel.INFO, sink=self.log.emit)

            await self.agent.connect()
            self.connected.emit(True)
            self.log.emit("连接成功：Playwright MCP 已就绪")

            while True:
                item = await self.q.get()
                if item is None:
                    break
                kind, payload = item

                if kind == "chat":
                    text = await self.agent.chat(payload)
                    self.reply.emit(text or "")
                elif kind == "status":
                    self.status.emit(self.agent.status())
                elif kind == "clear":
                    self.agent.clear()
                    self.log.emit("已清空状态")
                elif kind == "save":
                    self.agent.save_session()
                    self.log.emit("已保存会话")

        except Exception as e:
            self.connected.emit(False)
            self.error.emit(str(e))
            self.log.emit(traceback.format_exc())
        finally:
            try:
                if self.agent:
                    await self.agent.disconnect()
            except Exception:
                pass
            self.connected.emit(False)

    def submit(self, kind: str, payload=None):
        if not self.loop or not self.q:
            return
        asyncio.run_coroutine_threadsafe(self.q.put((kind, payload)), self.loop)

    def stop(self):
        if self.loop and self.q:
            asyncio.run_coroutine_threadsafe(self.q.put(None), self.loop)


# ----------------------------
# Main Window
# ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: Optional[AgentWorker] = None

        self.setWindowTitle("Playwright MCP 桌面助手")
        self.resize(1120, 820)

        self._build_ui()
        self._apply_qss()
        self._set_connected(False)

        # 初始提示（也可只用横幅）
        self.add_system(
            "欢迎使用 Playwright MCP 桌面助手。"
           # "建议流程：输入 API Key → 点击【连接】 → 在下方输入任务（Ctrl+Enter 发送）。"
        )

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- 顶部 AppBar
        appbar = QtWidgets.QFrame()
        appbar.setObjectName("AppBar")
        appbar_l = QtWidgets.QHBoxLayout(appbar)
        appbar_l.setContentsMargins(14, 12, 14, 12)
        appbar_l.setSpacing(10)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        self.lbl_title = QtWidgets.QLabel("Playwright MCP")
        self.lbl_title.setObjectName("Title")
        self.lbl_sub = QtWidgets.QLabel("桌面助手 · PySide6 · 现代聊天界面")
        self.lbl_sub.setObjectName("SubTitle")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_sub)

        self.pill_status = QtWidgets.QLabel("未连接")
        self.pill_status.setObjectName("PillBad")

        appbar_l.addLayout(title_box)
        appbar_l.addSpacing(10)
        appbar_l.addWidget(self.pill_status, 0, QtCore.Qt.AlignVCenter)
        appbar_l.addStretch(1)

        self.apiKey = QtWidgets.QLineEdit()
        self.apiKey.setPlaceholderText("输入DeepSeek API Key(可留空:读取 config.json / 环境变量)")
        self.apiKey.setEchoMode(QtWidgets.QLineEdit.Password)
        self.apiKey.setMinimumWidth(380)
        self.apiKey.setObjectName("KeyInput")

        self.btnConnect = QtWidgets.QPushButton("连接")
        self.btnStatus = QtWidgets.QPushButton("状态")
        self.btnClear = QtWidgets.QPushButton("清空")
        self.btnSave = QtWidgets.QPushButton("保存")

        for b in (self.btnConnect, self.btnStatus, self.btnClear, self.btnSave):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setMinimumHeight(36)
            b.setObjectName("Btn")
        self.btnConnect.setObjectName("BtnPrimary")

        self.btnConnect.clicked.connect(self.on_connect)
        self.btnStatus.clicked.connect(self.on_status)
        self.btnClear.clicked.connect(self.on_clear)
        self.btnSave.clicked.connect(self.on_save)

        appbar_l.addWidget(self.apiKey)
        appbar_l.addWidget(self.btnConnect)
        appbar_l.addWidget(self.btnStatus)
        appbar_l.addWidget(self.btnClear)
        appbar_l.addWidget(self.btnSave)

        root.addWidget(appbar)

        # --- 聊天面板
        chat_panel = QtWidgets.QFrame()
        chat_panel.setObjectName("Panel")
        chat_l = QtWidgets.QVBoxLayout(chat_panel)
        chat_l.setContentsMargins(14, 14, 14, 14)
        chat_l.setSpacing(10)

        # 提示横幅（你要的“怎么操作”）
        self.hint = QtWidgets.QFrame()
        self.hint.setObjectName("HintBanner")
        hint_l = QtWidgets.QHBoxLayout(self.hint)
        hint_l.setContentsMargins(12, 10, 12, 10)
        hint_l.setSpacing(10)

        icon = QtWidgets.QLabel("使用提示")
        icon.setObjectName("HintIcon")

        text = QtWidgets.QLabel(
            "1) 输入 API Key（或放到 config.json / 环境变量）\n"
            "2) 点击【连接】等待成功\n"
            "3) 在下方输入任务，例如：打开百度\n"
            "快捷键：Ctrl + Enter 发送"
        )
        text.setObjectName("HintText")
        text.setWordWrap(True)

        hint_l.addWidget(icon, 0, QtCore.Qt.AlignTop)
        hint_l.addWidget(text, 1)

        chat_l.addWidget(self.hint)

        # 聊天列表
        self.model = ChatModel()
        self.view = QtWidgets.QListView()
        self.view.setModel(self.model)
        self.view.setItemDelegate(BubbleDelegate(self.view))
        self.view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setObjectName("ChatView")
        self.view.setSpacing(6)

        chat_l.addWidget(self.view, 1)
        root.addWidget(chat_panel, 1)

        # --- 输入区
        composer = QtWidgets.QFrame()
        composer.setObjectName("Composer")
        comp_l = QtWidgets.QHBoxLayout(composer)
        comp_l.setContentsMargins(14, 12, 14, 12)
        comp_l.setSpacing(10)

        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText("输入任务…（Ctrl+Enter 发送）\n例如：打开百度 / 打开 github.com 搜索 python")
        self.input.setFixedHeight(116)
        self.input.setObjectName("ComposerInput")

        self.btnSend = QtWidgets.QPushButton("发送")
        self.btnSend.setCursor(QtCore.Qt.PointingHandCursor)
        self.btnSend.setMinimumHeight(44)
        self.btnSend.setMinimumWidth(120)
        self.btnSend.setObjectName("BtnPrimary")
        self.btnSend.clicked.connect(self.on_send)

        comp_l.addWidget(self.input, 1)
        comp_l.addWidget(self.btnSend)
        root.addWidget(composer)

        # 快捷键：Ctrl+Enter 发送
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self.on_send)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self.on_send)

    def _apply_qss(self):
        self.setStyleSheet(f"""
        QMainWindow {{
            background: {Palette.BG};
        }}

        /* AppBar */
        QFrame#AppBar {{
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {Palette.PANEL}, stop:1 {Palette.PANEL_2});
            border: 1px solid {Palette.BORDER};
            border-radius: 16px;
        }}
        QLabel#Title {{
            color: {Palette.TEXT};
            font-size: 18px;
            font-weight: 800;
        }}
        QLabel#SubTitle {{
            color: {Palette.MUTED};
            font-size: 11px;
        }}

        /* Status pill */
        QLabel#PillGood, QLabel#PillBad {{
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
        }}
        QLabel#PillGood {{
            color: rgba(255,255,255,0.95);
            background: rgba(53,198,107,0.18);
            border: 1px solid rgba(53,198,107,0.35);
        }}
        QLabel#PillBad {{
            color: rgba(255,255,255,0.88);
            background: rgba(255,92,92,0.16);
            border: 1px solid rgba(255,92,92,0.32);
        }}

        /* Inputs & Buttons */
        QLineEdit#KeyInput {{
            color: {Palette.TEXT};
            background: rgba(255,255,255,0.06);
            border: 1px solid {Palette.BORDER};
            border-radius: 12px;
            padding: 9px 12px;
            selection-background-color: {Palette.ACCENT_2};
        }}
        QLineEdit#KeyInput:focus {{
            border: 1px solid rgba(121,168,255,0.60);
            background: rgba(121,168,255,0.08);
        }}

        QPushButton#Btn, QPushButton#BtnPrimary {{
            border-radius: 12px;
            padding: 8px 12px;
            border: 1px solid {Palette.BORDER};
            background: rgba(255,255,255,0.05);
            color: {Palette.TEXT};
        }}
        QPushButton#Btn:hover {{
            background: rgba(255,255,255,0.09);
            border: 1px solid {Palette.BORDER_2};
        }}
        QPushButton#BtnPrimary {{
            background: rgba(121,168,255,0.22);
            border: 1px solid rgba(121,168,255,0.45);
            font-weight: 800;
        }}
        QPushButton#BtnPrimary:hover {{
            background: rgba(121,168,255,0.30);
            border: 1px solid rgba(121,168,255,0.60);
        }}
        QPushButton:disabled {{
            color: rgba(255,255,255,0.38);
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
        }}

        /* Panels */
        QFrame#Panel {{
            background: {Palette.PANEL};
            border: 1px solid {Palette.BORDER};
            border-radius: 16px;
        }}

        /* Hint banner */
        QFrame#HintBanner {{
            background: rgba(121,168,255,0.12);
            border: 1px solid rgba(121,168,255,0.26);
            border-radius: 14px;
        }}
        QLabel#HintIcon {{
            color: rgba(255,255,255,0.92);
            background: rgba(121,168,255,0.24);
            border: 1px solid rgba(121,168,255,0.40);
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 900;
        }}
        QLabel#HintText {{
            color: rgba(255,255,255,0.84);
            font-size: 11px;
            line-height: 1.35;
        }}

        /* Chat view */
        QListView#ChatView {{
            background: transparent;
            border: none;
            outline: none;
            color: rgba(255,255,255,0.92);
        }}
        QListView#ChatView::item {{
            border: none;
        }}

        /* Composer */
        QFrame#Composer {{
            background: {Palette.PANEL};
            border: 1px solid {Palette.BORDER};
            border-radius: 16px;
        }}
        QPlainTextEdit#ComposerInput {{
            color: {Palette.TEXT};
            background: rgba(255,255,255,0.06);
            border: 1px solid {Palette.BORDER};
            border-radius: 14px;
            padding: 12px 12px;
            selection-background-color: {Palette.ACCENT_2};
        }}
        QPlainTextEdit#ComposerInput:focus {{
            border: 1px solid rgba(121,168,255,0.60);
            background: rgba(121,168,255,0.08);
        }}

        /* Scrollbar */
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(255,255,255,0.16);
            border-radius: 6px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(255,255,255,0.24);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        """)

    def _scroll_to_bottom(self):
        QtCore.QTimer.singleShot(0, lambda: self.view.scrollToBottom())

    def add_user(self, text: str):
        self.model.add(ChatMessage(role=MsgRole.USER, text=text, ts=now_hhmm()))
        self._scroll_to_bottom()

    def add_assistant_pending(self):
        self.model.add(ChatMessage(role=MsgRole.ASSISTANT, text="我正在处理你的请求…", ts=now_hhmm(), pending=True))
        self._scroll_to_bottom()

    def set_assistant_reply(self, text: str):
        # 如果没有 pending（极端情况），就直接追加
        before = len(self.model.items)
        self.model.update_last_pending(new_text=text, pending=False)
        after = len(self.model.items)
        if before == after:
            self.model.add(ChatMessage(role=MsgRole.ASSISTANT, text=text, ts=now_hhmm(), pending=False))
        self._scroll_to_bottom()

    def add_log(self, text: str):
        self.model.add(ChatMessage(role=MsgRole.LOG, text=text, ts=now_hhmm()))
        self._scroll_to_bottom()

    def add_system(self, text: str):
        self.model.add(ChatMessage(role=MsgRole.SYSTEM, text=text, ts=now_hhmm()))
        self._scroll_to_bottom()

    def _set_connected(self, ok: bool):
        self.btnConnect.setEnabled(not ok)
        self.btnSend.setEnabled(ok)
        self.btnStatus.setEnabled(ok)
        self.btnClear.setEnabled(ok)
        self.btnSave.setEnabled(ok)

        if ok:
            self.pill_status.setText("已连接")
            self.pill_status.setObjectName("PillGood")
        else:
            self.pill_status.setText("未连接")
            self.pill_status.setObjectName("PillBad")

        self.pill_status.style().unpolish(self.pill_status)
        self.pill_status.style().polish(self.pill_status)

    # ----------------------------
    # Actions
    # ----------------------------
    def on_connect(self):
        key = self.apiKey.text().strip() or Config.get_api_key()
        if not key:
            self.add_system("未检测到 API Key：请在上方输入，或写入 config.json / 环境变量 DEEPSEEK_API_KEY。")
            return

        if self.apiKey.text().strip():
            Config.set_api_key(key)

        self.worker = AgentWorker(key)
        self.worker.connected.connect(self._set_connected)
        self.worker.log.connect(self.add_log)
        self.worker.status.connect(lambda t: self.add_system(t))
        self.worker.reply.connect(self.set_assistant_reply)
        self.worker.error.connect(lambda e: self.add_system("连接/运行异常： " + e))

        self.add_log("正在连接 Playwright MCP（首次可能较慢）…")
        self.worker.start()

    def on_send(self):
        if not self.worker:
            self.add_system("未连接：请先点击【连接】。")
            return

        msg = self.input.toPlainText().strip()
        if not msg:
            return

        self.input.clear()
        self.add_user(msg)
        self.add_assistant_pending()
        self.worker.submit("chat", msg)

    def on_status(self):
        if self.worker:
            self.worker.submit("status")

    def on_clear(self):
        if self.worker:
            self.worker.submit("clear")
        self.model.clear()
        self.add_system("已清空对话与状态。")

    def on_save(self):
        if self.worker:
            self.worker.submit("save")
        self.add_system("已触发保存会话。")

    def closeEvent(self, e):
        try:
            if self.worker:
                self.worker.stop()
                self.worker.wait(1500)
        except Exception:
            pass
        super().closeEvent(e)


def main():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QtWidgets.QApplication(sys.argv)

    # Windows 上更舒服的字体
    font = QtGui.QFont("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
