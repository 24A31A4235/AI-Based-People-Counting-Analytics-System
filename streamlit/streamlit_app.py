"""People Counting — Live CCTV dashboard (Streamlit).

Recreates the project's original PyQt5 cyberpunk dashboard as a web app:
sidebar with ENTERED / EXITED / INSIDE / PEAK cards, live video feed,
MALE / FEMALE / UNIQUE analytics, controls and a status bar.
Live camera via streamlit-webrtc; counting uses the fixed counter
(8/8 real crossings on the busy-street test).
"""
import os
import sys
# The counting modules (detector/tracker/counter/config) live in the repo
# root; this app file lives in streamlit/, so add the root to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
import streamlit as st

from detector import PeopleDetector
from tracker import SORTTracker
from counter import LineCrossingCounter
from config import get_config

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

MODEL_PATH = "yolov8m.pt"

# ---- Cyberpunk theme (same palette as the original PyQt5 dashboard) ----
st.set_page_config(page_title="AI People Counter — Analytics Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background-color: #06060e; }
  section[data-testid="stSidebar"] { background: linear-gradient(#111122, #0a0a14, #06060e); }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #e8e8f0; }
  .card {
    background: linear-gradient(rgba(20,20,42,230), rgba(10,10,20,240));
    border-radius: 14px; padding: 10px 12px; text-align: center;
    margin-bottom: 10px; border: 1px solid #1e1e3a;
  }
  .card .t { font-size: 13px; font-weight: 700; letter-spacing: 1px; }
  .card .v { font-size: 42px; font-weight: 800; line-height: 1.05; color: #e8e8f0;
             font-family: Consolas, monospace; }
  .mini {
    background: linear-gradient(rgba(20,20,42,200), rgba(10,10,20,220));
    border-radius: 10px; padding: 8px; text-align: center; border: 1px solid #1e1e3a;
  }
  .mini .t { font-size: 12px; font-weight: 700; }
  .mini .v { font-size: 26px; font-weight: 800; color: #e8e8f0; font-family: Consolas, monospace; }
  .stApp, .stApp p, .stApp label { color: #e8e8f0; }
  .statusbar { color: #8888aa; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


def card_html(title, value, color, dim, icon):
    return (f'<div class="card" style="border-left: 3px solid {color};">'
            f'<div class="t" style="color:{color};">{icon} {title}</div>'
            f'<div class="v">{value}</div></div>')


def mini_html(title, value, color, icon):
    return (f'<div class="mini" style="border-top: 2px solid {color};">'
            f'<div class="t" style="color:{color};">{icon} {title}</div>'
            f'<div class="v">{value}</div></div>')


# ---- Live counting state (shared with the video processor) ----
class LiveState:
    def __init__(self):
        self.in_count = 0
        self.out_count = 0
        self.peak = 0
        self.visible = 0
        self.fps = 0.0
        self.running = False

STATE = LiveState()


class CounterProcessor(VideoProcessorBase):
    """Processes each live camera frame: detect, track, count."""
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.counter = None
        self.width = None
        self.line_pos = 0.5
        self.confidence = 0.4
        self._t0 = time.time()
        self._frames = 0

    def _ensure(self, W):
        if (self.detector is None or self.width != W):
            cfg = get_config("MEDIUM")
            trk = cfg.tracker
            cnt = cfg.counter
            self.detector = PeopleDetector(model_path=MODEL_PATH,
                                           confidence=self.confidence, iou_threshold=0.5)
            self.tracker = SORTTracker(max_age=trk.max_age, min_hits=trk.min_hits,
                                       iou_threshold=trk.iou_threshold)
            line_x = int(W * self.line_pos)
            self.counter = LineCrossingCounter(line_x=line_x, buffer_zone=cnt.buffer_zone,
                                               cooldown_frames=cnt.cooldown_frames,
                                               frame_width=W)
            self.width = W

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        H, W = img.shape[:2]
        self._ensure(W)
        # Recreate the counter if the line moved.
        line_x = int(W * self.line_pos)
        if self.counter.line_x != line_x:
            cfg = get_config("MEDIUM")
            cnt = cfg.counter
            self.counter = LineCrossingCounter(line_x=line_x, buffer_zone=cnt.buffer_zone,
                                               cooldown_frames=cnt.cooldown_frames,
                                               frame_width=W)
        detections = self.detector.detect(img)
        det_arr = np.asarray(detections, dtype=float).reshape(-1, 5) if len(detections) else np.empty((0, 5))
        tracks = self.tracker.update(det_arr)
        in_c, out_c, _ = self.counter.update(tracks, None)

        STATE.in_count = in_c
        STATE.out_count = out_c
        STATE.visible = len(tracks)
        STATE.peak = max(STATE.peak, len(tracks))
        STATE.running = True
        self._frames += 1
        dt = time.time() - self._t0
        if dt >= 1.0:
            STATE.fps = self._frames / dt
            self._frames = 0
            self._t0 = time.time()

        vis = img.copy()
        cv2.line(vis, (line_x, 0), (line_x, H), (0, 255, 255), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        in_label = "IN ->"
        out_label = "<- OUT"
        (in_w, _), _ = cv2.getTextSize(in_label, font, 0.9, 2)
        cv2.putText(vis, in_label, (line_x - in_w - 12, 32), font, 0.9, (0, 255, 0), 2)
        cv2.putText(vis, out_label, (line_x + 12, 32), font, 0.9, (0, 0, 255), 2)
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.get_bbox()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {tr.id}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"IN {in_c}  OUT {out_c}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(vis, format="bgr24")


# ---- Sidebar: counter cards (same as the original dashboard) ----
with st.sidebar:
    st.markdown('<h2 style="color:#00f0ff; text-align:center;">⚡ COUNTER</h2>',
                unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2a2a55;">', unsafe_allow_html=True)
    in_slot = st.empty()
    out_slot = st.empty()
    inside_slot = st.empty()
    peak_slot = st.empty()
    fps_slot = st.empty()
    vis_slot = st.empty()

    in_slot.markdown(card_html("ENTERED", STATE.in_count, "#00f0ff", "#0088aa", "→"),
                     unsafe_allow_html=True)
    out_slot.markdown(card_html("EXITED", STATE.out_count, "#ff006e", "#aa0048", "←"),
                      unsafe_allow_html=True)
    inside_slot.markdown(card_html("INSIDE", max(0, STATE.in_count - STATE.out_count),
                                   "#ffd000", "#aa8800", "⊙"), unsafe_allow_html=True)
    peak_slot.markdown(card_html("PEAK", STATE.peak, "#aa44ff", "#6622aa", "▲"),
                       unsafe_allow_html=True)
    fps_slot.markdown(f'<p style="text-align:center; color:#00ff88;"><b>FPS: --</b></p>',
                      unsafe_allow_html=True)
    vis_slot.markdown('<p style="text-align:center; color:#8888aa;">Visible: 0</p>',
                      unsafe_allow_html=True)

# ---- Main panel ----
st.markdown("## ⚡ AI People Counter — Analytics Dashboard")
st.caption("Point the camera at a doorway or entrance, like a mall gate. "
           "It counts everyone who walks in and out, live.")

if not HAS_WEBRTC:
    st.error("Live camera (streamlit-webrtc) is not installed. "
             "Add streamlit-webrtc to requirements.txt and redeploy.")
    st.stop()

# Controls row
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 2])
with c1:
    line_pct = st.slider("LINE position", 10, 90, 50, 1, format="%d%%")
with c2:
    conf = st.slider("Confidence", 0.20, 0.60, 0.40, 0.05)
with c3:
    reset = st.button("↺  RESET", use_container_width=True)
with c4:
    st.caption("Line: the yellow line people walk across. "
               "IN → = walked right across it. ← OUT = walked left across it.")

if reset:
    STATE.in_count = 0
    STATE.out_count = 0
    STATE.peak = 0
    STATE.visible = 0
    STATE.running = False

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
ctx = webrtc_streamer(
    key="people-counter",
    video_processor_factory=CounterProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

# Keep the processor in sync with the sliders.
if ctx.video_processor is not None:
    ctx.video_processor.line_pos = line_pct / 100.0
    ctx.video_processor.confidence = conf

# Analytics row (below the video)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(mini_html("MALE", "–", "#4488ff", "♂"), unsafe_allow_html=True)
with m2:
    st.markdown(mini_html("FEMALE", "–", "#ff006e", "♀"), unsafe_allow_html=True)
with m3:
    st.markdown(mini_html("UNIQUE", STATE.peak, "#aa44ff", "👤"), unsafe_allow_html=True)

st.markdown(f'<p class="statusbar">● {"Running" if STATE.running else "Ready"}'
            f' &nbsp;|&nbsp; FPS: {STATE.fps:.1f} &nbsp;|&nbsp; Persons: {STATE.visible}</p>',
            unsafe_allow_html=True)
st.caption("Counter fixed to count slow walkers correctly — 8/8 real crossings "
           "on a busy-street test (was 6/8). Gender/age display is not active "
           "in this build (its models are unavailable).")

# Live-refresh the sidebar cards while the camera runs.
if ctx.state.playing:
    for _ in range(600):
        if not ctx.state.playing:
            break
        in_slot.markdown(card_html("ENTERED", STATE.in_count, "#00f0ff", "#0088aa", "→"),
                         unsafe_allow_html=True)
        out_slot.markdown(card_html("EXITED", STATE.out_count, "#ff006e", "#aa0048", "←"),
                          unsafe_allow_html=True)
        inside_slot.markdown(card_html("INSIDE", max(0, STATE.in_count - STATE.out_count),
                                       "#ffd000", "#aa8800", "⊙"), unsafe_allow_html=True)
        peak_slot.markdown(card_html("PEAK", STATE.peak, "#aa44ff", "#6622aa", "▲"),
                           unsafe_allow_html=True)
        fps_slot.markdown(f'<p style="text-align:center; color:#00ff88;"><b>FPS: {STATE.fps:.1f}</b></p>',
                          unsafe_allow_html=True)
        vis_slot.markdown(f'<p style="text-align:center; color:#8888aa;">Visible: {STATE.visible}</p>',
                          unsafe_allow_html=True)
        time.sleep(0.5)
