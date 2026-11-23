# NOTE: `spaces` MUST be imported first, before gradio/torch/ultralytics.
# Otherwise CUDA is initialised first and ZeroGPU mode refuses to start.
import spaces

import os
import cv2
import numpy as np
import gradio as gr

from detector import PeopleDetector
from tracker import SORTTracker
from counter import LineCrossingCounter
from config import get_config

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8m.pt")

# Live state, kept across frames. One camera feed = one counter.
_state = {"detector": None, "tracker": None, "counter": None,
          "width": None, "line_pos": None, "confidence": None}


def _reset_state():
    _state.update({"detector": None, "tracker": None, "counter": None,
                   "width": None, "line_pos": None, "confidence": None})


def _ensure_state(W, line_pos, confidence):
    """Create (or recreate) the detector/tracker/counter when the
    frame size, line position or confidence changes."""
    if (_state["detector"] is None or _state["width"] != W
            or _state["line_pos"] != line_pos or _state["confidence"] != confidence):
        cfg = get_config("MEDIUM")
        trk_cfg = cfg.tracker
        cnt_cfg = cfg.counter
        _state["detector"] = PeopleDetector(model_path=MODEL_PATH, confidence=confidence,
                                            iou_threshold=0.5)
        _state["tracker"] = SORTTracker(max_age=trk_cfg.max_age, min_hits=trk_cfg.min_hits,
                                        iou_threshold=trk_cfg.iou_threshold)
        line_x = int(W * line_pos)
        _state["counter"] = LineCrossingCounter(line_x=line_x, buffer_zone=cnt_cfg.buffer_zone,
                                                cooldown_frames=cnt_cfg.cooldown_frames,
                                                frame_width=W)
        _state["width"] = W
        _state["line_pos"] = line_pos
        _state["confidence"] = confidence


@spaces.GPU
def process_frame(frame, line_pos, confidence):
    """Run one live camera frame through detection + tracking + counting.
    Called continuously while the webcam streams. Returns the annotated
    frame and the live IN / OUT counts."""
    if frame is None:
        return None, "0", "0"
    # Gradio gives RGB; OpenCV works in BGR.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    _ensure_state(W, line_pos, confidence)

    detector = _state["detector"]
    tracker = _state["tracker"]
    counter = _state["counter"]
    line_x = int(W * line_pos)

    detections = detector.detect(bgr)
    det_arr = np.asarray(detections, dtype=float).reshape(-1, 5) if len(detections) else np.empty((0, 5))
    tracks = tracker.update(det_arr)
    in_c, out_c, _ = counter.update(tracks, None)

    vis = bgr.copy()
    cv2.line(vis, (line_x, 0), (line_x, H), (0, 255, 255), 2)
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.get_bbox()]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"ID {tr.id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, f"IN {in_c}  OUT {out_c}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return rgb, str(in_c), str(out_c)


def reset_counts():
    _reset_state()
    return None, "0", "0"


with gr.Blocks(title="People Counting — Live CCTV") as demo:
    gr.Markdown(
        "# People Counting — Live CCTV\n"
        "Point your camera at a doorway or entrance, like a mall gate. "
        "The app watches the live feed and counts everyone who walks in and out. "
        "**IN** = left to right, **OUT** = right to left.\n\n"
        "Allow camera access when the browser asks. The counting line is the yellow "
        "line — set the slider so it sits across the doorway people walk through."
    )
    with gr.Row():
        cam_in = gr.Image(sources=["webcam"], streaming=True, label="Live camera")
        cam_out = gr.Image(label="Counted live feed (people boxed, line in yellow)")
    with gr.Row():
        line_pos = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                             label="Counting line position (fraction of width)")
        confidence = gr.Slider(0.2, 0.6, value=0.4, step=0.05,
                               label="Detection confidence (lower = finds more people)")
    with gr.Row():
        in_box = gr.Textbox(label="IN (entered)", interactive=False, value="0")
        out_box = gr.Textbox(label="OUT (exited)", interactive=False, value="0")
    reset_btn = gr.Button("Reset counts")
    gr.Markdown(
        "How it works: **YOLOv8** finds people in each frame · **SORT tracker** "
        "gives each person a stable ID · the **line-crossing counter** counts a "
        "person only when they are tracked, settle on one side, and move clearly "
        "past the line — so jitter and brief glimpses are not counted. "
        "The counter was fixed to count slow walkers correctly (8/8 real crossings "
        "on a busy-street test, up from 6/8).\n\n"
        "Tip: mount the camera so people walk *across* the yellow line, not along it. "
        "If someone walks along the line instead of across it, they will not be counted.\n\n"
        "Model: YOLOv8m (person class only). Code: github.com/24A31A4235/AI-Based-People-Counting-Analytics-System"
    )
    cam_in.stream(process_frame,
                  inputs=[cam_in, line_pos, confidence],
                  outputs=[cam_out, in_box, out_box])
    reset_btn.click(reset_counts, inputs=[], outputs=[cam_out, in_box, out_box])

if __name__ == "__main__":
    demo.launch()
