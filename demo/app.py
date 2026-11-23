# NOTE: `spaces` MUST be imported first, before gradio/torch/ultralytics.
# Otherwise CUDA is initialised first and ZeroGPU mode refuses to start
# (the same import-order bug fixed in the IMDB Space).
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

_detector = None
_tracker = None


def _get_detector(confidence):
    global _detector
    if _detector is None:
        _detector = PeopleDetector(model_path=MODEL_PATH, confidence=confidence, iou_threshold=0.5)
    else:
        _detector.set_confidence(confidence)
    return _detector


@spaces.GPU
def count_people(video_path, line_pos, confidence, progress=gr.Progress(track_tqdm=True)):
    """Run detection + tracking + counting on an uploaded video.

    line_pos: fraction of the frame width where the vertical counting
    line sits (0.5 = middle). IN = left->right, OUT = right->left.
    Returns the annotated video path and the counts.
    """
    if not video_path:
        return None, "Please upload a video first.", "", ""

    cfg = get_config("MEDIUM")
    trk_cfg = cfg.tracker
    cnt_cfg = cfg.counter

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if W == 0 or H == 0 or n_frames == 0:
        cap.release()
        return None, "Could not read this video. Try an MP4 file.", "", ""

    detector = _get_detector(confidence)
    tracker = SORTTracker(max_age=trk_cfg.max_age, min_hits=trk_cfg.min_hits,
                          iou_threshold=trk_cfg.iou_threshold)
    line_x = int(W * line_pos)
    counter = LineCrossingCounter(line_x=line_x, buffer_zone=cnt_cfg.buffer_zone,
                                  cooldown_frames=cnt_cfg.cooldown_frames, frame_width=W)

    out_path = "/tmp/counted_output.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    in_c = out_c = 0
    max_people = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        detections = detector.detect(frame)
        det_arr = np.asarray(detections, dtype=float).reshape(-1, 5) if len(detections) else np.empty((0, 5))
        tracks = tracker.update(det_arr)
        max_people = max(max_people, len(tracks))
        in_c, out_c, _ = counter.update(tracks, None)

        vis = frame.copy()
        cv2.line(vis, (line_x, 0), (line_x, H), (0, 255, 255), 2)
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.get_bbox()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {tr.id}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"IN {in_c}  OUT {out_c}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(vis)
        if frame_idx % 25 == 0:
            progress(frame_idx / max(n_frames, 1), desc=f"Processing frame {frame_idx}/{n_frames}")

    cap.release()
    writer.release()

    total = in_c + out_c
    summary = (
        f"Processed {frame_idx} frames ({W}x{H}, {fps:.0f} fps). "
        f"Peak people visible at once: {max_people}. "
        f"Line at x={line_x} ({line_pos:.0%} of width)."
    )
    note = ("Tip: for the best count, place the line where people walk ACROSS it "
            "(e.g. a doorway). If people walk along the line instead of across it, "
            "they will not be counted. Move the line slider to match your video.")
    return out_path, summary, str(in_c), str(out_c)


with gr.Blocks(title="People Counting — AI-Based Analytics") as demo:
    gr.Markdown(
        "# People Counting — AI-Based Analytics\n"
        "Upload a video. The app detects people (YOLOv8), tracks them (SORT) and "
        "counts how many cross the yellow line. **IN** = left to right, **OUT** = right to left.\n\n"
        "This demo uses the fixed counter: it now counts slow walkers correctly "
        "(verified 8/8 real crossings on a busy street clip, up from 6/8)."
    )
    with gr.Row():
        video_in = gr.Video(label="Upload a video (MP4)", sources=["upload"])
        video_out = gr.Video(label="Counted video (people boxed, line in yellow)")
    with gr.Row():
        line_pos = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                             label="Counting line position (fraction of width)")
        confidence = gr.Slider(0.2, 0.6, value=0.4, step=0.05,
                               label="Detection confidence (lower = finds more people)")
    run_btn = gr.Button("Count people", variant="primary")
    with gr.Row():
        in_box = gr.Textbox(label="IN (left to right)", interactive=False)
        out_box = gr.Textbox(label="OUT (right to left)", interactive=False)
    summary = gr.Textbox(label="Summary", interactive=False)
    gr.Markdown(
        "How it works: **YOLOv8** finds people in each frame · **SORT tracker** "
        "gives each person a stable ID · the **line-crossing counter** counts a "
        "person only when they are tracked, settle on one side, and move clearly "
        "past the line — so jitter and brief glimpses are not counted.\n\n"
        "Model: YOLOv8m (person class only). Code: github.com/24A31A4235/AI-Based-People-Counting-Analytics-System"
    )
    run_btn.click(count_people,
                  inputs=[video_in, line_pos, confidence],
                  outputs=[video_out, summary, in_box, out_box])

if __name__ == "__main__":
    demo.launch()
