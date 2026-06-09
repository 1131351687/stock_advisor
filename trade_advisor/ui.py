"""
A股策略决策系统 — PySide6 图形界面
"""

import sys
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTextEdit, QLineEdit, QMessageBox, QSplitter,
    QFrame, QAbstractItemView, QDialog, QDialogButtonBox,
    QGridLayout, QDateEdit, QCheckBox, QMenu, QToolBar, QStatusBar,
    QInputDialog, QStyle, QScrollArea, QFileDialog,
)

import hikyuu
from hikyuu.interactive import *
import numpy as np

from trade_advisor.strategies import REGISTERED_STRATEGIES, StrategyParam, StrategyResult
from trade_advisor.storage import (
    init_db, save_decision, save_decision_batch, save_decision_log,
    get_decisions, get_decisions_by_date, update_decision,
    delete_decision, clear_decisions,
)
from trade_advisor.backtest import run_backtest, BacktestResult
from trade_advisor.strategies import _get_last_trade_date

# matplotlib 嵌入 Qt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── 样式 ──
STYLESHEET = """
QMainWindow {
    background-color: #f5f6fa;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #dcdde1;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    background: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QTableWidget {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    background: white;
    gridline-color: #ecf0f1;
    selection-background-color: #3498db;
}
QHeaderView::section {
    background-color: #2c3e50;
    color: white;
    padding: 6px;
    border: none;
    font-weight: bold;
}
QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2980b9;
}
QPushButton:pressed {
    background-color: #2471a3;
}
QPushButton.danger {
    background-color: #e74c3c;
}
QPushButton.danger:hover {
    background-color: #c0392b;
}
QPushButton.success {
    background-color: #27ae60;
}
QPushButton.success:hover {
    background-color: #229954;
}
QComboBox {
    padding: 4px 8px;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background: white;
    min-width: 150px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 4px;
    background: white;
}
QTabWidget::pane {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    background: white;
}
QTabBar::tab {
    background: #ecf0f1;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: white;
    border-bottom: 2px solid #3498db;
    font-weight: bold;
}
QStatusBar {
    background: #2c3e50;
    color: white;
}
"""


# ── 参数控件工厂 ──
def create_param_widget(param_def: StrategyParam, value=None):
    """根据参数定义创建对应的输入控件"""
    v = value if value is not None else param_def.default

    if param_def.param_type == "int":
        w = QSpinBox()
        w.setRange(int(param_def.min_val or 1), int(param_def.max_val or 999))
        w.setValue(int(v))
        return w
    elif param_def.param_type == "float":
        w = QDoubleSpinBox()
        w.setRange(param_def.min_val or 0, param_def.max_val or 1e9)
        w.setSingleStep(param_def.step or 0.1)
        w.setDecimals(2)
        w.setValue(float(v))
        return w
    elif param_def.param_type == "choice" and param_def.options:
        w = QComboBox()
        for opt in param_def.options:
            w.addItem(str(opt), str(opt))
        idx = w.findData(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return w
    else:
        w = QLineEdit(str(v))
        return w


def get_param_value(widget):
    """从控件获取参数值"""
    if isinstance(widget, QSpinBox):
        return widget.value()
    elif isinstance(widget, QDoubleSpinBox):
        return widget.value()
    elif isinstance(widget, QComboBox):
        return widget.currentData()
    elif isinstance(widget, QLineEdit):
        return widget.text().strip()
    return None


# ── 添加决策对话框 ──
class DecisionDialog(QDialog):
    """添加/编辑决策记录的对话框"""

    def __init__(self, parent=None, record=None):
        super().__init__(parent)
        self.record = record
        self.setWindowTitle("编辑决策" if record else "添加决策")
        self.setMinimumWidth(500)
        self.setup_ui()
        if record:
            self.load_record()

    def setup_ui(self):
        layout = QFormLayout(self)
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(datetime.now().date())
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        layout.addRow("日期:", self.date_edit)

        self.strategy_combo = QComboBox()
        for key, s in REGISTERED_STRATEGIES.items():
            self.strategy_combo.addItem(s.name, key)
        layout.addRow("策略:", self.strategy_combo)

        self.action_combo = QComboBox()
        self.action_combo.addItems(["buy", "sell", "hold"])
        layout.addRow("操作:", self.action_combo)

        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("如 SZ002001")
        layout.addRow("股票代码:", self.code_edit)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("股票名称")
        layout.addRow("股票名称:", self.name_edit)

        self.price_edit = QDoubleSpinBox()
        self.price_edit.setRange(0, 99999)
        self.price_edit.setDecimals(2)
        self.price_edit.setPrefix("¥ ")
        layout.addRow("价格:", self.price_edit)

        self.shares_edit = QSpinBox()
        self.shares_edit.setRange(0, 99999999)
        self.shares_edit.setSingleStep(100)
        layout.addRow("数量:", self.shares_edit)

        self.reason_edit = QLineEdit()
        self.reason_edit.setPlaceholderText("决策理由")
        layout.addRow("理由:", self.reason_edit)

        self.notes_edit = QLineEdit()
        layout.addRow("备注:", self.notes_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def load_record(self):
        r = self.record
        from datetime import datetime as dt
        self.date_edit.setDate(dt.strptime(r["date"], "%Y-%m-%d").date())
        idx = self.strategy_combo.findText(r["strategy"])
        if idx >= 0:
            self.strategy_combo.setCurrentIndex(idx)
        ai = self.action_combo.findText(r["action"])
        if ai >= 0:
            self.action_combo.setCurrentIndex(ai)
        self.code_edit.setText(r.get("code", ""))
        self.name_edit.setText(r.get("name", ""))
        self.price_edit.setValue(float(r.get("price", 0)))
        self.shares_edit.setValue(int(r.get("shares", 0)))
        self.reason_edit.setText(r.get("reason", ""))
        self.notes_edit.setText(r.get("notes", ""))

    def get_data(self) -> dict:
        return {
            "date": self.date_edit.date().toString("yyyy-MM-dd"),
            "strategy": self.strategy_combo.currentText(),
            "action": self.action_combo.currentText(),
            "code": self.code_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "price": self.price_edit.value(),
            "shares": self.shares_edit.value(),
            "reason": self.reason_edit.text().strip(),
            "notes": self.notes_edit.text().strip(),
        }


# ── 主窗口 ──
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A股策略决策系统")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(STYLESHEET)

        # 当前策略结果
        self.current_result = None
        self.current_bt_result = None

        self.setup_statusbar()
        self.setup_ui()
        self.setup_timer()

        # 初始化数据库
        init_db()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(8)

        # ── 顶部工具栏 ──
        self.create_toolbar(main_layout)

        # ── 主区域 Tab ──
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)

        self.tab_today = QWidget()
        self.tab_history = QWidget()
        self.tab_backtest = QWidget()
        self.tabs.addTab(self.tab_today, "📋 今日决策")
        self.tabs.addTab(self.tab_backtest, "📈 回测分析")
        self.tabs.addTab(self.tab_history, "📜 历史记录")

        self.setup_today_tab()
        self.setup_backtest_tab()
        self.setup_history_tab()

    def create_toolbar(self, parent_layout):
        toolbar = QFrame()
        toolbar.setFrameShape(QFrame.StyledPanel)
        toolbar.setStyleSheet("background: #2c3e50; border-radius: 6px; padding: 6px;")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 6, 12, 6)

        title = QLabel("📊 A股策略决策系统")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        tl.addWidget(title)

        tl.addStretch()

        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumWidth(140)
        self.strategy_combo.setStyleSheet("background: white;")
        for key, s in REGISTERED_STRATEGIES.items():
            self.strategy_combo.addItem(s.name, key)
        self.strategy_combo.currentIndexChanged.connect(self.on_strategy_changed)
        tl.addWidget(QLabel("策略:"))
        tl.addWidget(self.strategy_combo)

        tl.addWidget(QLabel("日期:"))
        self.trade_date_edit = QDateEdit()
        self.trade_date_edit.setCalendarPopup(True)
        self.trade_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.trade_date_edit.setDate(datetime.now().date())
        self.trade_date_edit.setStyleSheet("background: white;")
        # 默认显示最后一个交易日
        try:
            _cal = sm.get_trading_calendar(Query(-30))
            if _cal and len(_cal) > 0:
                _last = _cal[-1]
                _dt = _last.datetime() if hasattr(_last, 'datetime') else _last
                self.trade_date_edit.setDate(_dt.date())
        except Exception:
            pass
        tl.addWidget(self.trade_date_edit)

        self.run_btn = QPushButton("▶ 执行选股")
        self.run_btn.setStyleSheet("background: #27ae60; color: white; font-weight: bold; padding: 6px 20px;")
        self.run_btn.clicked.connect(self.on_run_strategy)
        tl.addWidget(self.run_btn)

        parent_layout.addWidget(toolbar)

    def setup_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

    def setup_timer(self):
        """定时刷新状态"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(60000)  # 每分钟

    def update_status(self):
        pass  # 预留

    # ── 今日决策 Tab ──
    def setup_today_tab(self):
        layout = QVBoxLayout(self.tab_today)
        layout.setSpacing(8)

        # 策略说明
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #7f8c8d; padding: 4px;")
        layout.addWidget(self.desc_label)

        # 参数 + 持仓区域(水平排列)
        params_hold_row = QHBoxLayout()

        # 参数区域
        self.param_group = QGroupBox("策略参数")
        self.param_layout = QGridLayout(self.param_group)
        self.param_widgets = {}
        params_hold_row.addWidget(self.param_group, 2)

        # 当前持仓区域
        hold_group = QGroupBox("当前持仓")
        hold_layout = QVBoxLayout(hold_group)
        self.holdings_edit = QTextEdit()
        self.holdings_edit.setPlaceholderText(
            "输入持仓股票代码，每行一个\n如: SZ002830\n    SZ003003\n\n留空则只显示买入信号"
        )
        self.holdings_edit.setMaximumHeight(80)
        hold_layout.addWidget(self.holdings_edit)

        load_hold_btn = QPushButton("加载上次决策")
        load_hold_btn.setStyleSheet("background: #95a5a6; color: white; padding: 4px 12px; font-size: 11px;")
        load_hold_btn.clicked.connect(self.load_holdings_from_history)
        hold_layout.addWidget(load_hold_btn)

        params_hold_row.addWidget(hold_group, 1)
        layout.addLayout(params_hold_row)

        # 结果显示
        result_group = QGroupBox("选股结果")
        rl = QVBoxLayout(result_group)

        # 信息行
        info_row = QHBoxLayout()
        self.result_info = QLabel("点击「执行选股」开始分析")
        self.result_info.setStyleSheet("color: #7f8c8d;")
        info_row.addWidget(self.result_info)
        info_row.addStretch()

        self.save_btn = QPushButton("💾 保存决策")
        self.save_btn.setStyleSheet("background: #27ae60; color: white;")
        self.save_btn.clicked.connect(self.on_save_decision)
        self.save_btn.setEnabled(False)
        info_row.addWidget(self.save_btn)

        rl.addLayout(info_row)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(8)
        self.result_table.setHorizontalHeaderLabels(
            ["#", "代码", "名称", "信号", "流通市值(亿)", "价格", "权重", "理由"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rl.addWidget(self.result_table)

        layout.addWidget(result_group, 1)

        # 初始化策略参数
        self.on_strategy_changed()

    def on_strategy_changed(self):
        """策略切换时更新参数面板"""
        key = self.strategy_combo.currentData()
        strategy = REGISTERED_STRATEGIES.get(key)
        if not strategy:
            return

        self.desc_label.setText(f"📌 {strategy.description}")

        # 清空旧参数
        for w in self.param_widgets.values():
            w.setParent(None)
        self.param_widgets.clear()

        # 移除旧布局中的所有行
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # 添加新参数
        for i, p in enumerate(strategy.params):
            label = QLabel(p.label + ":")
            widget = create_param_widget(p)
            self.param_layout.addWidget(label, i // 3, (i % 3) * 2)
            self.param_layout.addWidget(widget, i // 3, (i % 3) * 2 + 1)
            self.param_widgets[p.name] = widget

        # 清空结果
        self.result_table.setRowCount(0)
        self.result_info.setText("点击「执行选股」开始分析")
        self.save_btn.setEnabled(False)
        self.current_result = None

    def on_run_strategy(self):
        """执行策略"""
        key = self.strategy_combo.currentData()
        strategy = REGISTERED_STRATEGIES.get(key)
        if not strategy:
            return

        # 收集参数
        params = {}
        for p in strategy.params:
            w = self.param_widgets.get(p.name)
            if w:
                params[p.name] = get_param_value(w)

        # 读取当前持仓
        holdings_text = self.holdings_edit.toPlainText().strip()
        holdings_list = []
        if holdings_text:
            for line in holdings_text.split("\n"):
                code = line.strip().upper()
                if code:
                    holdings_list.append(code)

        # 获取所选日期
        selected_date = self.trade_date_edit.date().toString("yyyy-MM-dd")
        self.status.showMessage(f"正在执行策略 ({selected_date})...")
        QApplication.processEvents()

        try:
            self.current_result = strategy.run(params, date=selected_date, holdings=holdings_list or None)
            self.display_result(self.current_result)
            self.status.showMessage(f"策略完成 — 交易日: {self.current_result.date}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "策略错误", f"运行出错: {e}")
            self.status.showMessage("策略执行失败", 5000)

    def display_result(self, result: StrategyResult):
        """在表格中显示结果（含买入/卖出/持有）"""
        self.result_table.setRowCount(0)
        all_signals = result.signals
        buy_signals = [s for s in all_signals if s.signal == "buy"]

        if not all_signals:
            err = result.metadata.get("error", "未选出符合条件股票")
            self.result_info.setText(f"⚠ {err}")
            self.save_btn.setEnabled(False)
            return

        self.result_table.setRowCount(len(all_signals))
        for i, sig in enumerate(all_signals):
            self.result_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.result_table.setItem(i, 1, QTableWidgetItem(sig.code))
            self.result_table.setItem(i, 2, QTableWidgetItem(sig.name))

            signal_text = {"buy": "🟢 买入", "sell": "🔴 卖出", "hold": "⏸ 持有"} \
                .get(sig.signal, sig.signal)
            signal_item = QTableWidgetItem(signal_text)
            if sig.signal == "buy":
                signal_item.setForeground(QColor("#27ae60"))
            elif sig.signal == "sell":
                signal_item.setForeground(QColor("#e74c3c"))
            elif sig.signal == "hold":
                signal_item.setForeground(QColor("#f39c12"))
            self.result_table.setItem(i, 3, signal_item)

            cap_item = QTableWidgetItem(f"{sig.market_cap / 1e8:.2f}" if sig.market_cap else "-")
            self.result_table.setItem(i, 4, cap_item)

            self.result_table.setItem(i, 5, QTableWidgetItem(f"{sig.price:.2f}" if sig.price else "-"))
            self.result_table.setItem(i, 6, QTableWidgetItem(f"{sig.weight:.1%}" if sig.weight else "-"))
            self.result_table.setItem(i, 7, QTableWidgetItem(sig.reason))

        # 汇总信息
        n_buy = sum(1 for s in all_signals if s.signal == "buy")
        n_sell = sum(1 for s in all_signals if s.signal == "sell")
        n_hold = sum(1 for s in all_signals if s.signal == "hold")
        meta = result.metadata
        info_parts = [f"交易日: {result.date}",
                     f"买入: {n_buy}只", f"卖出: {n_sell}只", f"持有: {n_hold}只",
                     f"候选: {meta.get('total_candidates', '-')}只"]
        self.result_info.setText(" | ".join(info_parts))
        self.save_btn.setEnabled(n_buy > 0)

    def on_save_decision(self):
        """保存本次决策到历史"""
        if not self.current_result:
            return

        # 保存本次决策的所有信号（买入+持有+卖出）
        all_signals = self.current_result.signals
        buy_signals = [s for s in all_signals if s.signal == "buy"]

        decisions = []
        for sig in all_signals:
            decisions.append({
                "date": self.current_result.date,
                "strategy": self.current_result.strategy_name,
                "action": sig.signal,  # 保留原始信号: buy/hold/sell
                "code": sig.code,
                "name": sig.name,
                "price": sig.price,
                "shares": 0,
                "reason": sig.reason,
                "notes": "",
            })

        if decisions:
            save_decision_batch(decisions)
        save_decision_log(
            self.current_result.date,
            self.current_result.strategy_name,
            {
                "signals": [s.__dict__ for s in all_signals],
                "metadata": self.current_result.metadata,
            }
        )

        n_buy = sum(1 for s in all_signals if s.signal == "buy")
        n_sell = sum(1 for s in all_signals if s.signal == "sell")
        n_hold = sum(1 for s in all_signals if s.signal == "hold")
        QMessageBox.information(self, "保存成功",
                                f"买入 {n_buy} 只 | 卖出 {n_sell} 只 | 持有 {n_hold} 只\n"
                                f"日期: {self.current_result.date}  策略: {self.current_result.strategy_name}")

        # 保存后自动更新持仓为本次买入+持有
        new_codes = [sig.code for sig in all_signals if sig.signal in ("buy", "hold")]
        if new_codes:
            self.holdings_edit.setPlainText("\n".join(new_codes))

        self.refresh_history()
        self.status.showMessage(f"已保存 {len(hold_signals)} 条持仓", 3000)

    # ── 回测分析 Tab ──
    def setup_backtest_tab(self):
        layout = QVBoxLayout(self.tab_backtest)
        layout.setSpacing(8)

        # 设置区域
        config_group = QGroupBox("回测设置")
        cfg = QGridLayout(config_group)

        cfg.addWidget(QLabel("策略:"), 0, 0)
        self.bt_strategy = QComboBox()
        for key, s in REGISTERED_STRATEGIES.items():
            self.bt_strategy.addItem(s.name, key)
        cfg.addWidget(self.bt_strategy, 0, 1)

        cfg.addWidget(QLabel("起始:"), 0, 2)
        self.bt_start = QDateEdit()
        self.bt_start.setCalendarPopup(True)
        self.bt_start.setDisplayFormat("yyyy-MM-dd")
        self.bt_start.setDate(datetime.now().date().replace(year=datetime.now().year - 1))
        cfg.addWidget(self.bt_start, 0, 3)

        cfg.addWidget(QLabel("结束:"), 0, 4)
        self.bt_end = QDateEdit()
        self.bt_end.setCalendarPopup(True)
        self.bt_end.setDisplayFormat("yyyy-MM-dd")
        self.bt_end.setDate(datetime.now().date())
        cfg.addWidget(self.bt_end, 0, 5)

        cfg.addWidget(QLabel("本金:"), 0, 6)
        self.bt_cash = QSpinBox()
        self.bt_cash.setRange(10000, 99999999)
        self.bt_cash.setValue(100000)
        self.bt_cash.setSingleStep(50000)
        self.bt_cash.setPrefix("¥ ")
        cfg.addWidget(self.bt_cash, 0, 7)

        cfg.addWidget(QLabel("调仓周期:"), 1, 0)
        self.bt_rebalance = QComboBox()
        self.bt_rebalance.addItem("每日", 1)
        self.bt_rebalance.addItem("每7天(周)", 7)
        self.bt_rebalance.addItem("每14天(2周)", 14)
        self.bt_rebalance.addItem("每30天(月)", 30)
        self.bt_rebalance.addItem("每90天(季)", 90)
        cfg.addWidget(self.bt_rebalance, 1, 1)

        # 参数
        self.bt_params_widgets = {}
        self.bt_params_layout = QGridLayout()
        bt_params_box = QGroupBox("策略参数")
        bt_params_box.setLayout(self.bt_params_layout)

        def on_bt_strategy_changed():
            key = self.bt_strategy.currentData()
            s = REGISTERED_STRATEGIES.get(key)
            if not s:
                return
            for w in self.bt_params_widgets.values():
                w.setParent(None)
            self.bt_params_widgets.clear()
            while self.bt_params_layout.count():
                item = self.bt_params_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            for i, p in enumerate(s.params):
                lbl = QLabel(p.label + ":")
                w = create_param_widget(p)
                self.bt_params_layout.addWidget(lbl, i // 4, (i % 4) * 2)
                self.bt_params_layout.addWidget(w, i // 4, (i % 4) * 2 + 1)
                self.bt_params_widgets[p.name] = w

        self.bt_strategy.currentIndexChanged.connect(on_bt_strategy_changed)

        cfg.addWidget(bt_params_box, 2, 0, 1, 8)

        self.bt_run_btn = QPushButton("▶ 运行回测")
        self.bt_run_btn.setStyleSheet("background: #e67e22; color: white; font-weight: bold; padding: 8px 24px;")
        self.bt_run_btn.clicked.connect(self.on_run_backtest)
        cfg.addWidget(self.bt_run_btn, 3, 0, 1, 8)

        layout.addWidget(config_group)

        # 结果区域放在 QScrollArea 中
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        result_group = QGroupBox("回测结果")
        rl = QVBoxLayout(result_group)
        result_group = QGroupBox("回测结果")
        rl = QVBoxLayout(result_group)

        # 指标行
        metrics_grid = QGridLayout()
        self.bt_metrics = {}
        for i, (key, label) in enumerate([
            ("total_return", "总收益率"), ("annual_return", "年化收益率"),
            ("max_drawdown", "最大回撤"), ("sharpe", "夏普比率"),
            ("trade_count", "交易次数"), ("trading_days", "回测天数"),
        ]):
            lbl = QLabel(f"{label}:")
            val = QLabel("--")
            val.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
            metrics_grid.addWidget(lbl, i // 3, (i % 3) * 2)
            metrics_grid.addWidget(val, i // 3, (i % 3) * 2 + 1)
            self.bt_metrics[key] = val

        rl.addLayout(metrics_grid)

        # 图表（matplotlib）
        self.bt_figure = Figure(figsize=(8, 3), dpi=100)
        self.bt_canvas = FigureCanvasQTAgg(self.bt_figure)
        self.bt_ax1 = self.bt_figure.add_subplot(121)
        self.bt_ax2 = self.bt_figure.add_subplot(122)
        rl.addWidget(self.bt_canvas)

        # 月度收益和交易列表用 Tab
        bt_detail = QTabWidget()
        self.bt_monthly_table = QTableWidget()
        self.bt_monthly_table.setColumnCount(3)
        self.bt_monthly_table.setHorizontalHeaderLabels(["月份", "月度收益", "累计收益"])
        self.bt_monthly_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bt_detail.addTab(self.bt_monthly_table, "月度收益")

        self.bt_trade_table = QTableWidget()
        self.bt_trade_table.setColumnCount(9)
        self.bt_trade_table.setHorizontalHeaderLabels(
            ["日期", "操作", "代码", "名称", "价格", "数量", "金额", "持仓数", "理由"]
        )
        self.bt_trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 交易明细工具栏：导出按钮
        trade_toolbar = QHBoxLayout()
        self.bt_export_btn = QPushButton("📥 导出CSV")
        self.bt_export_btn.clicked.connect(self.on_export_bt_trades)
        self.bt_export_btn.setStyleSheet("background: #8e44ad; color: white; padding: 4px 12px; font-size: 11px;")
        trade_toolbar.addStretch()
        trade_toolbar.addWidget(self.bt_export_btn)

        trade_tab_widget = QWidget()
        trade_tab_layout = QVBoxLayout(trade_tab_widget)
        trade_tab_layout.setContentsMargins(0, 0, 0, 0)
        trade_tab_layout.addLayout(trade_toolbar)
        trade_tab_layout.addWidget(self.bt_trade_table)
        bt_detail.addTab(trade_tab_widget, "交易明细")

        rl.addWidget(bt_detail, 1)
        scroll_layout.addWidget(result_group, 1)

        # 初始化参数
        on_bt_strategy_changed()

    def on_run_backtest(self):
        key = self.bt_strategy.currentData()
        start = self.bt_start.date().toString("yyyy-MM-dd")
        end = self.bt_end.date().toString("yyyy-MM-dd")
        cash = self.bt_cash.value()

        params = {}
        s = REGISTERED_STRATEGIES.get(key)
        if s:
            for p in s.params:
                w = self.bt_params_widgets.get(p.name)
                if w:
                    params[p.name] = get_param_value(w)

        self.status.showMessage("正在回测，请稍候...")
        QApplication.processEvents()

        rebal = self.bt_rebalance.currentData()
        try:
            result = run_backtest(key, start, end, params, cash, rebalance_days=rebal)
            self.current_bt_result = result
            self.display_backtest_results(result)
            self.status.showMessage(f"回测完成 — {result.start_date} ~ {result.end_date}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "回测错误", f"运行出错: {e}")
            self.status.showMessage("回测失败", 5000)

    def display_backtest_results(self, result: BacktestResult):
        # 指标
        color_g = "#27ae60"
        color_r = "#e74c3c"

        def set_metric(key, val_str, color=color_g):
            if key in self.bt_metrics:
                self.bt_metrics[key].setText(val_str)
                self.bt_metrics[key].setStyleSheet(
                    f"font-size: 16px; font-weight: bold; color: {color};"
                )

        set_metric("total_return", f"{result.total_return_pct:+.2f}%",
                   color_g if result.total_return_pct >= 0 else color_r)
        set_metric("annual_return", f"{result.annual_return_pct:+.2f}%",
                   color_g if result.annual_return_pct >= 0 else color_r)
        set_metric("max_drawdown", f"{result.max_drawdown_pct:.2f}%", color_r)
        set_metric("sharpe", f"{result.sharpe_ratio:.2f}",
                   color_g if result.sharpe_ratio >= 1 else "#f39c12")
        set_metric("trade_count", str(result.trade_count))
        set_metric("trading_days", str(result.trading_days))

        # 图表
        self.bt_figure.clear()
        ax1 = self.bt_figure.add_subplot(121)
        ax2 = self.bt_figure.add_subplot(122)

        if result.nav_series is not None and len(result.nav_series) > 0:
            idx = pd.to_datetime(result.nav_series.index)
            ax1.plot(idx, result.nav_series.values, color="#2980b9", linewidth=1.2)
            ax1.axhline(y=result.init_cash, color="gray", linestyle="--", alpha=0.5)
            ax1.set_title("资产曲线", fontsize=10)
            ax1.set_ylabel("总资产")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=30)

        if result.drawdown_series is not None and len(result.drawdown_series) > 0:
            idx = pd.to_datetime(result.drawdown_series.index)
            ax2.fill_between(idx, result.drawdown_series.values, 0,
                             color="#e74c3c", alpha=0.3)
            ax2.set_title("回撤曲线", fontsize=10)
            ax2.set_ylabel("回撤(%)")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=30)

        self.bt_figure.tight_layout()
        self.bt_canvas.draw()

        # 月度收益
        self.bt_monthly_table.setRowCount(0)
        months = result.monthly_returns
        if months:
            self.bt_monthly_table.setRowCount(len(months))
            cum = 100
            for i, (period, ret) in enumerate(sorted(months.items())):
                cum *= (1 + ret / 100)
                self.bt_monthly_table.setItem(i, 0, QTableWidgetItem(str(period)))
                ret_item = QTableWidgetItem(f"{ret:+.2f}%")
                ret_item.setForeground(QColor(color_g if ret >= 0 else color_r))
                self.bt_monthly_table.setItem(i, 1, ret_item)
                self.bt_monthly_table.setItem(i, 2, QTableWidgetItem(f"{cum - 100:+.2f}%"))

        # 交易明细
        self.bt_trade_table.setRowCount(0)
        trades = result.trades
        if trades:
            self.bt_trade_table.setRowCount(min(len(trades), 500))
            shown = trades[-500:] if len(trades) > 500 else trades
            for i, t in enumerate(shown):
                self.bt_trade_table.setItem(i, 0, QTableWidgetItem(t.date))
                act_item = QTableWidgetItem({"buy": "🟢 买入", "sell": "🔴 卖出"}.get(t.action, t.action))
                act_item.setForeground(QColor(color_g if t.action == "buy" else color_r))
                self.bt_trade_table.setItem(i, 1, act_item)
                self.bt_trade_table.setItem(i, 2, QTableWidgetItem(t.code))
                self.bt_trade_table.setItem(i, 3, QTableWidgetItem(t.name))
                self.bt_trade_table.setItem(i, 4, QTableWidgetItem(f"{t.price:.2f}"))
                self.bt_trade_table.setItem(i, 5, QTableWidgetItem(str(t.shares)))
                self.bt_trade_table.setItem(i, 6, QTableWidgetItem(f"{t.value:,.2f}"))
                self.bt_trade_table.setItem(i, 7, QTableWidgetItem(f"{t.holdings_count}只" if t.holdings_count else "-"))
                self.bt_trade_table.setItem(i, 8, QTableWidgetItem(t.reason))

    def on_export_bt_trades(self):
        """导出回测交易记录到CSV"""
        result = self.current_bt_result
        if not result or not result.trades:
            QMessageBox.information(self, "提示", "暂无回测数据可导出")
            return
        from pathlib import Path
        default_name = f"回测交易_{result.start_date}_{result.end_date}.csv"
        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出交易记录", str(Path.home() / default_name),
            "CSV文件 (*.csv)"
        )
        if not fpath:
            return
        try:
            import csv
            with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["日期", "操作", "代码", "名称", "价格", "数量", "金额", "持仓数", "理由"])
                for t in result.trades:
                    w.writerow([
                        t.date,
                        {"buy": "买入", "sell": "卖出"}.get(t.action, t.action),
                        t.code, t.name, t.price, t.shares, round(t.value, 2),
                        t.holdings_count, t.reason,
                    ])
            self.status.showMessage(f"已导出 {len(result.trades)} 条交易到 {fpath}", 5000)
            QMessageBox.information(self, "导出成功", f"已导出 {len(result.trades)} 条记录\n{fpath}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def load_holdings_from_history(self):
        """从最近一次策略决策中加载持仓（读取 buy+hold 信号）"""
        from trade_advisor.storage import get_decision_logs
        logs = get_decision_logs(limit=1)
        if not logs:
            QMessageBox.information(self, "提示", "暂无历史决策记录")
            return
        try:
            data = json.loads(logs[0].get("full_result", "{}"))
            signals = data.get("signals", [])
            # 加载 buy 和 hold 信号（两者都表示应持有的股票）
            codes = [s.get("code", "") for s in signals if s.get("signal") in ("buy", "hold")]
            # 也查一下 decisions 表作为备选
            if not codes:
                from trade_advisor.storage import get_decisions
                recents = get_decisions(limit=10)
                codes = list(dict.fromkeys([r["code"] for r in recents if r.get("code")]))
            if codes:
                self.holdings_edit.setPlainText("\n".join(codes))
                self.status.showMessage(f"已加载 {len(codes)} 只持仓", 3000)
            else:
                QMessageBox.information(self, "提示", "上次决策无持仓记录")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载失败: {e}")

    # ── 历史记录方法 ──
    def setup_history_tab(self):
        layout = QVBoxLayout(self.tab_history)
        layout.setSpacing(8)

        # 操作栏
        action_row = QHBoxLayout()

        self.date_filter = QDateEdit()
        self.date_filter.setCalendarPopup(True)
        self.date_filter.setDate(datetime.now().date())
        self.date_filter.setDisplayFormat("yyyy-MM-dd")
        self.date_filter.setSpecialValueText("全部")
        action_row.addWidget(QLabel("日期:"))
        action_row.addWidget(self.date_filter)

        self.filter_btn = QPushButton("查询")
        self.filter_btn.clicked.connect(self.refresh_history)
        action_row.addWidget(self.filter_btn)

        self.show_all_btn = QPushButton("显示全部")
        self.show_all_btn.clicked.connect(
            lambda: (self.date_filter.setDate(datetime.now().date()),
                     self.date_filter.setSpecialValueText("全部"),
                     self.refresh_history())
        )
        action_row.addWidget(self.show_all_btn)

        action_row.addStretch()

        self.add_btn = QPushButton("＋ 添加")
        self.add_btn.clicked.connect(self.on_add_decision)
        action_row.addWidget(self.add_btn)

        self.edit_btn = QPushButton("✎ 编辑")
        self.edit_btn.clicked.connect(self.on_edit_decision)
        action_row.addWidget(self.edit_btn)

        self.del_btn = QPushButton("✕ 删除")
        self.del_btn.setStyleSheet("background: #e74c3c; color: white;")
        self.del_btn.clicked.connect(self.on_delete_decision)
        action_row.addWidget(self.del_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #bdc3c7;")
        action_row.addWidget(sep)

        self.view_detail_btn = QPushButton("📋 查看详情")
        self.view_detail_btn.clicked.connect(self.on_view_decision_detail)
        action_row.addWidget(self.view_detail_btn)

        self.export_history_btn = QPushButton("📥 导出CSV")
        self.export_history_btn.setStyleSheet("background: #8e44ad; color: white;")
        self.export_history_btn.clicked.connect(self.on_export_history)
        action_row.addWidget(self.export_history_btn)

        layout.addLayout(action_row)

        # 表格
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(10)
        self.history_table.setHorizontalHeaderLabels(
            ["ID", "日期", "策略", "操作", "代码", "名称", "价格", "数量", "理由", "备注"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.history_table, 1)

        # 刷新
        self.refresh_history()

    def refresh_history(self):
        """刷新历史记录表格"""
        val = self.date_filter.specialValueText()
        if val == "全部":
            records = get_decisions(limit=500)
        else:
            date_str = self.date_filter.date().toString("yyyy-MM-dd")
            records = get_decisions_by_date(date_str)

        self.history_table.setRowCount(len(records))
        for i, r in enumerate(records):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(r["id"])))
            self.history_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.history_table.setItem(i, 2, QTableWidgetItem(r["strategy"]))

            action_item = QTableWidgetItem(
                {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(r["action"], r["action"])
            )
            if r["action"] == "buy":
                action_item.setForeground(QColor("#27ae60"))
            elif r["action"] == "sell":
                action_item.setForeground(QColor("#e74c3c"))
            self.history_table.setItem(i, 3, action_item)

            self.history_table.setItem(i, 4, QTableWidgetItem(r.get("code", "")))
            self.history_table.setItem(i, 5, QTableWidgetItem(r.get("name", "")))
            self.history_table.setItem(i, 6, QTableWidgetItem(
                f"{r['price']:.2f}" if r.get("price") else "-"
            ))
            self.history_table.setItem(i, 7, QTableWidgetItem(
                str(r.get("shares", "")) if r.get("shares") else "-"
            ))
            self.history_table.setItem(i, 8, QTableWidgetItem(r.get("reason", "")))
            self.history_table.setItem(i, 9, QTableWidgetItem(r.get("notes", "")))

        self.status.showMessage(f"共 {len(records)} 条记录")

    def on_add_decision(self):
        dlg = DecisionDialog(self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            save_decision(**data)
            self.refresh_history()
            self.status.showMessage("记录已添加", 3000)

    def on_edit_decision(self):
        row = self.history_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        rid = int(self.history_table.item(row, 0).text())
        record = {
            "id": rid,
            "date": self.history_table.item(row, 1).text(),
            "strategy": self.history_table.item(row, 2).text(),
            "action": {"买入": "buy", "卖出": "sell", "持有": "hold"}
                      .get(self.history_table.item(row, 3).text(), "buy"),
            "code": self.history_table.item(row, 4).text(),
            "name": self.history_table.item(row, 5).text(),
            "price": float(self.history_table.item(row, 6).text().replace("¥", "") or 0),
            "shares": int(self.history_table.item(row, 7).text() or 0),
            "reason": self.history_table.item(row, 8).text(),
            "notes": self.history_table.item(row, 9).text(),
        }
        dlg = DecisionDialog(self, record)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            update_decision(rid, **data)
            self.refresh_history()
            self.status.showMessage("记录已更新", 3000)

    def on_delete_decision(self):
        rows = set()
        for item in self.history_table.selectedItems():
            rows.add(item.row())
        if not rows:
            QMessageBox.information(self, "提示", "请先选择要删除的记录")
            return

        if QMessageBox.question(self, "确认删除",
                                f"确定删除 {len(rows)} 条记录?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        for r in sorted(rows, reverse=True):
            rid = int(self.history_table.item(r, 0).text())
            delete_decision(rid)
        self.refresh_history()
        self.status.showMessage(f"已删除 {len(rows)} 条记录", 3000)

    def on_view_decision_detail(self):
        """查看历史决策的详细选股结果"""
        from trade_advisor.storage import get_decision_logs
        logs = get_decision_logs(limit=10)
        if not logs:
            QMessageBox.information(self, "提示", "暂无决策日志")
            return
        # 取最新的日志显示
        log = logs[0]
        try:
            data = json.loads(log.get("full_result", "{}"))
            signals = data.get("signals", [])
            if not signals:
                QMessageBox.information(self, "提示", "该决策无选股记录")
                return
            lines = [f"📅 {log['date']}  |  策略: {log['strategy']}",
                     "─" * 40]
            for s in signals:
                sig_text = {"buy": "🟢买入", "hold": "⏸持有", "sell": "🔴卖出"}.get(s.get("signal", ""), "")
                code = s.get("code", "")
                name = s.get("name", "")
                reason = s.get("reason", "")
                lines.append(f"  {sig_text} {code} {name}  — {reason}")
            meta = data.get("metadata", {})
            if meta:
                lines.append("")
                lines.append(f"交易日: {meta.get('trade_date', '')}")
                lines.append(f"候选: {meta.get('total_candidates', '-')}只")

            QMessageBox.information(self, f"决策详情 — {log['date']}",
                                    "\n".join(lines))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载失败: {e}")

    def on_export_history(self):
        """导出历史交易记录到CSV"""
        from trade_advisor.storage import get_decisions
        records = get_decisions(limit=5000)
        if not records:
            QMessageBox.information(self, "提示", "暂无历史记录可导出")
            return
        from pathlib import Path
        fpath, _ = QFileDialog.getSaveFileName(
            self, "导出历史记录", str(Path.home() / "历史交易记录.csv"),
            "CSV文件 (*.csv)"
        )
        if not fpath:
            return
        try:
            import csv
            with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["日期", "策略", "操作", "代码", "名称", "价格", "数量", "理由", "备注"])
                for r in records:
                    w.writerow([
                        r["date"], r["strategy"],
                        {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(r["action"], r["action"]),
                        r.get("code", ""), r.get("name", ""),
                        r.get("price", ""), r.get("shares", ""),
                        r.get("reason", ""), r.get("notes", ""),
                    ])
            self.status.showMessage(f"已导出 {len(records)} 条记录", 3000)
            QMessageBox.information(self, "导出成功", f"已导出 {len(records)} 条记录\n{fpath}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))


def run():
    # 初始化 Hikyuu
    try:
        q = Query(-1)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))

    # 设置字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
