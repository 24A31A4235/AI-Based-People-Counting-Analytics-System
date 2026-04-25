"""
ui.py - Premium Cyberpunk Dashboard UI for People Counting System
Built with PyQt5, featuring a glassmorphism sidebar + main content layout.

Layout:
┌──────────┬───────────────────────────────────┐
│  SIDEBAR │       LIVE VIDEO FEED             │
│          │                                   │
│ ┌──────┐ │                                   │
│ │ IN   │ │                                   │
│ │ 12   │ │                                   │
│ └──────┘ ├───────────┬───────────┬───────────┤
│ ┌──────┐ │  ♂ MALE   │ ♀ FEMALE │ 👤 UNIQUE │
│ │ OUT  │ │   5       │   7      │   10      │
│ │  3   │ ├───────────┴───────────┴───────────┤
│ └──────┘ │  [Source ▼] [▶ START] [🔄] [💾]   │
│ ┌──────┐ │  [━━━━━━━━ Line: 50% ━━━━━━━━━]  │
│ │INSIDE│ ├───────────────────────────────────┤
│ │  9   │ │  Status: Running  |  FPS: 28.5   │
│ └──────┘ └───────────────────────────────────┘
│ ┌──────┐ 
│ │ PEAK │ 
│ │  15  │ 
│ └──────┘ 
└──────────┘

Design: Deep space theme with neon accent colors and glassmorphism.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSlider, QFileDialog, QStatusBar,
    QSizePolicy, QSpacerItem, QGroupBox, QGridLayout, QComboBox,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon, QPainter, QLinearGradient


# =====================================================
# Color Palette — Cyberpunk / Deep Space Theme
# =====================================================
COLORS = {
    'bg_darkest':    '#06060e',
    'bg_dark':       '#0a0a14',
    'bg_medium':     '#111122',
    'bg_card':       '#14142a',
    'bg_card_hover': '#1a1a38',
    'border_subtle': '#1e1e3a',
    'border_glow':   '#2a2a55',
    'text_primary':  '#e8e8f0',
    'text_secondary':'#8888aa',
    'text_muted':    '#555577',
    # Neon accents
    'cyan':          '#00f0ff',
    'cyan_dim':      '#0088aa',
    'magenta':       '#ff006e',
    'magenta_dim':   '#aa0048',
    'gold':          '#ffd000',
    'gold_dim':      '#aa8800',
    'green':         '#00ff88',
    'green_dim':     '#00aa55',
    'blue':          '#4488ff',
    'blue_dim':      '#2255aa',
    'purple':        '#aa44ff',
    'purple_dim':    '#6622aa',
    'red':           '#ff3355',
    'red_dim':       '#aa1133',
}


class VideoDisplay(QLabel):
    """
    Custom QLabel for displaying video frames with premium styling.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 400)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_darkest']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 12px;
            }}
        """)
        self.setText("⏸  NO VIDEO FEED")
        self.setFont(QFont("Consolas", 14, QFont.Bold))
        self.setStyleSheet(self.styleSheet() + f"color: {COLORS['text_muted']};")

    def update_frame(self, frame):
        """Update the display with a new OpenCV frame."""
        if frame is None:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)


class GlowCard(QFrame):
    """
    A glassmorphic card widget with colored glow border.
    Used for counter displays in the sidebar.
    """
    def __init__(self, title, accent_color, accent_dim, icon="", parent=None):
        super().__init__(parent)
        self.accent = accent_color
        self.accent_dim = accent_dim
        self.setFixedHeight(100)
        
        self.setStyleSheet(f"""
            GlowCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(20, 20, 42, 230),
                    stop:1 rgba(10, 10, 20, 240));
                border: 1px solid {accent_dim};
                border-radius: 14px;
                border-left: 3px solid {accent_color};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        
        # Title row
        title_label = QLabel(f"{icon}  {title}")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title_label.setStyleSheet(f"color: {accent_color}; background: transparent; border: none;")
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel("0")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFont(QFont("Consolas", 36, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']}; background: transparent; border: none;")
        layout.addWidget(self.value_label)
    
    def set_value(self, value):
        """Update the counter value."""
        self.value_label.setText(str(value))


class MiniCard(QFrame):
    """
    A compact glassmorphic card for the analytics row below the video.
    """
    def __init__(self, title, accent_color, accent_dim, icon="", parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        
        self.setStyleSheet(f"""
            MiniCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(20, 20, 42, 200),
                    stop:1 rgba(10, 10, 20, 220));
                border: 1px solid {accent_dim};
                border-radius: 10px;
                border-top: 2px solid {accent_color};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(1)
        
        title_label = QLabel(f"{icon} {title}")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        title_label.setStyleSheet(f"color: {accent_color}; background: transparent; border: none;")
        layout.addWidget(title_label)
        
        self.value_label = QLabel("0")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFont(QFont("Consolas", 24, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']}; background: transparent; border: none;")
        layout.addWidget(self.value_label)
    
    def set_value(self, value):
        """Update the value display."""
        self.value_label.setText(str(value))


class PeopleCountingUI(QMainWindow):
    """
    Premium cyberpunk dashboard for the People Counting System.
    
    Features:
    - Sidebar with glowing counter cards
    - Main panel with video feed and analytics
    - Glassmorphic controls and status bar
    - Deep space color scheme with neon accents
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ AI People Counter — Analytics Dashboard")
        self.setMinimumSize(1050, 750)
        self.resize(1200, 850)

        # Apply theme
        self._apply_theme()

        # Build layout
        self._build_ui()

        # State
        self.is_running = False
        self.video_source = None
        self.line_position_ratio = 0.5

    def _apply_theme(self):
        """Apply deep space cyberpunk theme."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_darkest']};
            }}
            QWidget {{
                color: {COLORS['text_primary']};
                font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            }}
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 12px;
                font-weight: bold;
                min-width: 90px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_card_hover']};
                border-color: {COLORS['border_glow']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['bg_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_muted']};
                border-color: {COLORS['bg_medium']};
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['border_subtle']};
                height: 6px;
                background: {COLORS['bg_card']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['cyan']};
                border: 2px solid {COLORS['cyan_dim']};
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['cyan_dim']}, stop:1 {COLORS['cyan']});
                border-radius: 3px;
            }}
            QStatusBar {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border_subtle']};
                font-size: 11px;
                padding: 3px;
            }}
            QComboBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                min-width: 140px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 28px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['cyan_dim']};
                border: 1px solid {COLORS['border_subtle']};
            }}
        """)

    def _build_ui(self):
        """Build the premium sidebar + main content layout."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(8)

        # ==========================================
        # SIDEBAR (Left Panel)
        # ==========================================
        sidebar = QFrame()
        sidebar.setFixedWidth(175)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['bg_medium']},
                    stop:0.5 {COLORS['bg_dark']},
                    stop:1 {COLORS['bg_darkest']});
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 14px;
            }}
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 12, 8, 12)
        sidebar_layout.setSpacing(8)

        # Sidebar title
        logo_label = QLabel("⚡ COUNTER")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFont(QFont("Consolas", 13, QFont.Bold))
        logo_label.setStyleSheet(f"""
            color: {COLORS['cyan']};
            background: transparent;
            border: none;
            padding: 4px;
        """)
        sidebar_layout.addWidget(logo_label)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {COLORS['border_glow']}; border: none;")
        sidebar_layout.addWidget(sep)
        sidebar_layout.addSpacing(4)

        # IN Counter Card
        self.in_counter = GlowCard("ENTERED", COLORS['cyan'], COLORS['cyan_dim'], "→")
        sidebar_layout.addWidget(self.in_counter)

        # OUT Counter Card
        self.out_counter = GlowCard("EXITED", COLORS['magenta'], COLORS['magenta_dim'], "←")
        sidebar_layout.addWidget(self.out_counter)

        # INSIDE (Net) Counter Card
        self.net_card = GlowCard("INSIDE", COLORS['gold'], COLORS['gold_dim'], "⊙")
        sidebar_layout.addWidget(self.net_card)

        # PEAK Counter Card
        self.peak_card = GlowCard("PEAK", COLORS['purple'], COLORS['purple_dim'], "▲")
        sidebar_layout.addWidget(self.peak_card)

        sidebar_layout.addStretch()

        # FPS display in sidebar
        self.fps_sidebar = QLabel("FPS: --")
        self.fps_sidebar.setAlignment(Qt.AlignCenter)
        self.fps_sidebar.setFont(QFont("Consolas", 11, QFont.Bold))
        self.fps_sidebar.setStyleSheet(f"color: {COLORS['green']}; background: transparent; border: none;")
        sidebar_layout.addWidget(self.fps_sidebar)

        # Persons visible
        self.persons_sidebar = QLabel("Visible: 0")
        self.persons_sidebar.setAlignment(Qt.AlignCenter)
        self.persons_sidebar.setFont(QFont("Segoe UI", 10))
        self.persons_sidebar.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent; border: none;")
        sidebar_layout.addWidget(self.persons_sidebar)

        root_layout.addWidget(sidebar)

        # ==========================================
        # MAIN CONTENT (Right Panel)
        # ==========================================
        main_panel = QVBoxLayout()
        main_panel.setSpacing(8)

        # --- Video Display ---
        self.video_display = VideoDisplay()
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_panel.addWidget(self.video_display)

        # --- Analytics Row (below video) ---
        analytics_layout = QHBoxLayout()
        analytics_layout.setSpacing(8)

        self.male_card = MiniCard("MALE", COLORS['blue'], COLORS['blue_dim'], "♂")
        self.female_card = MiniCard("FEMALE", COLORS['magenta'], COLORS['magenta_dim'], "♀")
        self.unique_card = MiniCard("UNIQUE", COLORS['purple'], COLORS['purple_dim'], "👤")

        analytics_layout.addWidget(self.male_card)
        analytics_layout.addWidget(self.female_card)
        analytics_layout.addWidget(self.unique_card)

        main_panel.addLayout(analytics_layout)

        # --- Controls Row ---
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background: rgba(17, 17, 34, 200);
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 10px;
            }}
        """)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(10)

        # Source selection
        source_label = QLabel("SRC")
        source_label.setFont(QFont("Consolas", 10, QFont.Bold))
        source_label.setStyleSheet(f"color: {COLORS['text_muted']}; background: transparent; border: none;")
        controls_layout.addWidget(source_label)

        self.source_combo = QComboBox()
        self.source_combo.addItem("📷 Webcam 0", 0)
        self.source_combo.addItem("📷 Webcam 1", 1)
        self.source_combo.addItem("📁 Video File...", "file")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        controls_layout.addWidget(self.source_combo)

        controls_layout.addSpacerItem(QSpacerItem(12, 0, QSizePolicy.Fixed))

        # Start/Stop button
        self.start_btn = QPushButton("▶  START")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['green_dim']}, stop:1 #005530);
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 8px 24px;
                border-radius: 8px;
                border: 1px solid {COLORS['green']};
            }}
            QPushButton:hover {{
                background: {COLORS['green_dim']};
                border-color: {COLORS['green']};
            }}
            QPushButton:pressed {{
                background: #003320;
            }}
        """)
        controls_layout.addWidget(self.start_btn)

        # Reset button
        self.reset_btn = QPushButton("↺  RESET")
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_card']};
                color: {COLORS['gold']};
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 8px;
                border: 1px solid {COLORS['gold_dim']};
            }}
            QPushButton:hover {{ background: {COLORS['bg_card_hover']}; }}
            QPushButton:pressed {{ background: {COLORS['bg_dark']}; }}
        """)
        controls_layout.addWidget(self.reset_btn)

        # Save button
        self.save_btn = QPushButton("⬇  SAVE")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_card']};
                color: {COLORS['cyan']};
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 8px;
                border: 1px solid {COLORS['cyan_dim']};
            }}
            QPushButton:hover {{ background: {COLORS['bg_card_hover']}; }}
            QPushButton:pressed {{ background: {COLORS['bg_dark']}; }}
        """)
        controls_layout.addWidget(self.save_btn)

        controls_layout.addStretch()

        # Line position slider
        line_label = QLabel("LINE")
        line_label.setFont(QFont("Consolas", 10, QFont.Bold))
        line_label.setStyleSheet(f"color: {COLORS['text_muted']}; background: transparent; border: none;")
        controls_layout.addWidget(line_label)

        self.line_slider = QSlider(Qt.Horizontal)
        self.line_slider.setRange(10, 90)
        self.line_slider.setValue(50)
        self.line_slider.setFixedWidth(150)
        self.line_slider.setToolTip("Adjust counting line position")
        controls_layout.addWidget(self.line_slider)

        self.line_value_label = QLabel("50%")
        self.line_value_label.setFixedWidth(38)
        self.line_value_label.setFont(QFont("Consolas", 10))
        self.line_value_label.setStyleSheet(f"color: {COLORS['cyan']}; background: transparent; border: none;")
        self.line_slider.valueChanged.connect(
            lambda v: self.line_value_label.setText(f"{v}%")
        )
        controls_layout.addWidget(self.line_value_label)

        main_panel.addWidget(controls_frame)

        # --- Status Bar ---
        root_layout.addLayout(main_panel)

        # ==========================================
        # STATUS BAR
        # ==========================================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_indicator = QLabel("●")
        self.status_indicator.setFont(QFont("Segoe UI", 12))
        self.status_indicator.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.status_bar.addWidget(self.status_indicator)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_bar.addWidget(self.status_label, 1)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Consolas", 10, QFont.Bold))
        self.fps_label.setStyleSheet(f"color: {COLORS['green']};")
        self.status_bar.addPermanentWidget(self.fps_label)

        self.persons_label = QLabel("Persons: 0")
        self.persons_label.setFont(QFont("Segoe UI", 10))
        self.persons_label.setStyleSheet(f"color: {COLORS['blue']};")
        self.status_bar.addPermanentWidget(self.persons_label)

    def _on_source_changed(self, index):
        """Handle video source selection change."""
        data = self.source_combo.currentData()
        if data == "file":
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "",
                "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
            )
            if filepath:
                self.video_source = filepath
                self.source_combo.setItemText(index, f"📁 {filepath.split('/')[-1]}")
            else:
                self.source_combo.setCurrentIndex(0)
        else:
            self.video_source = data

    def update_counters(self, in_count, out_count):
        """Update the IN and OUT counter displays."""
        self.in_counter.set_value(in_count)
        self.out_counter.set_value(out_count)
        net = max(0, in_count - out_count)
        self.net_card.set_value(net)

    def update_fps(self, fps):
        """Update FPS displays."""
        fps_text = f"FPS: {fps:.1f}"
        self.fps_label.setText(fps_text)
        self.fps_sidebar.setText(fps_text)
        
        # Color code
        if fps >= 25:
            color = COLORS['green']
        elif fps >= 15:
            color = COLORS['gold']
        else:
            color = COLORS['red']
        self.fps_label.setStyleSheet(f"color: {color};")
        self.fps_sidebar.setStyleSheet(f"color: {color}; background: transparent; border: none;")

    def update_persons_count(self, count):
        """Update persons visible count."""
        self.persons_label.setText(f"Persons: {count}")
        self.persons_sidebar.setText(f"Visible: {count}")

    def update_gender_counts(self, male, female):
        """Update male/female displays."""
        self.male_card.set_value(male)
        self.female_card.set_value(female)

    def update_analytics(self, total_unique, peak_count):
        """Update analytics panel."""
        self.unique_card.set_value(total_unique)
        self.peak_card.set_value(peak_count)

    def update_status(self, text):
        """Update status bar text."""
        self.status_label.setText(text)

    def set_running_state(self, is_running):
        """Toggle UI state between running and stopped."""
        self.is_running = is_running
        if is_running:
            self.start_btn.setText("⏹  STOP")
            self.start_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {COLORS['red_dim']}, stop:1 #550011);
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 8px 24px;
                    border-radius: 8px;
                    border: 1px solid {COLORS['red']};
                }}
                QPushButton:hover {{ background: {COLORS['red_dim']}; }}
                QPushButton:pressed {{ background: #330008; }}
            """)
            self.status_indicator.setStyleSheet(f"color: {COLORS['green']};")
            self.update_status("Running")
            self.source_combo.setEnabled(False)
        else:
            self.start_btn.setText("▶  START")
            self.start_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {COLORS['green_dim']}, stop:1 #005530);
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 8px 24px;
                    border-radius: 8px;
                    border: 1px solid {COLORS['green']};
                }}
                QPushButton:hover {{ background: {COLORS['green_dim']}; }}
                QPushButton:pressed {{ background: #003320; }}
            """)
            self.status_indicator.setStyleSheet(f"color: {COLORS['red']};")
            self.update_status("Stopped")
            self.source_combo.setEnabled(True)

    def get_line_position_ratio(self):
        """Get the line position as a ratio (0.0 to 1.0)."""
        return self.line_slider.value() / 100.0

    def display_frame(self, frame):
        """Display a processed video frame."""
        self.video_display.update_frame(frame)
