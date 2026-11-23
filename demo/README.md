---
title: People Counting — Live CCTV
emoji: 🚶
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# People Counting — Live CCTV

A live people counter, like a CCTV camera at a mall entrance. Point your camera at a doorway and it counts everyone who walks in and out, in real time.

**IN** = left to right, **OUT** = right to left.

## How to use
1. Open the demo and allow camera access when the browser asks.
2. Point the camera at a doorway or entrance, so people walk **across** the frame.
3. Set the line slider so the yellow line sits across the doorway. People who cross it are counted; people who walk along the line are not.
4. Watch the IN / OUT counts update live. Use **Reset counts** to start a fresh count.

## What was fixed
The original counter undercounted: on a busy street test clip it counted **6 of 8** real line crossings. Two people were missed — the only right-to-left crosser, and a slow walker who only went 42px past the line.

Root causes found by running the pipeline on a real video:
- The counter required a person to be **52.5px** past the line. Slow walkers who cross but turn back soon were rejected. Now **28px**.
- The position history was capped at **60 frames**. For a slow walker (~1px/frame) the last 60 positions are all near the line, so the "was clearly on the old side" check failed. Now **600 frames**.

After the fix the same clip counts **8/8** real crossings, with no false counts. Detection itself was fine all along (YOLO finds ~14 people/frame; the smoothing filter only drops ~2%).

## Stack
YOLOv8m (person class) · SORT tracker (Kalman + Hungarian) · line-crossing counter · Gradio · ZeroGPU

Code: https://github.com/24A31A4235/AI-Based-People-Counting-Analytics-System
