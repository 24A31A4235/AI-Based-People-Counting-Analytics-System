---
title: People Counting — AI-Based Analytics
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

# People Counting — AI-Based Analytics

Upload a video. The app detects people (YOLOv8), tracks them (SORT) and counts how many cross the yellow line. **IN** = left to right, **OUT** = right to left.

## What was fixed
The original counter undercounted: on a busy street test clip it counted **6 of 8** real line crossings. Two people were missed — the only right-to-left crosser, and a slow walker who only went 42px past the line.

Root causes found by running the pipeline on a real video:
- The counter required a person to be **52.5px** past the line. Slow walkers who cross but turn back soon were rejected. Now **28px**.
- The position history was capped at **60 frames**. For a slow walker (~1px/frame) the last 60 positions are all near the line, so the "was clearly on the old side" check failed. Now **600 frames**.

After the fix the same clip counts **8/8** real crossings, with no false counts. Detection itself was fine all along (YOLO finds ~14 people/frame; the smoothing filter only drops ~2%).

## How to use
1. Upload an MP4 video of people walking.
2. Set the line position slider so the yellow line sits where people walk **across** it (e.g. a doorway). If people walk along the line instead of across it, they will not be counted.
3. Click **Count people**. You get an annotated video plus IN / OUT counts.

## Stack
YOLOv8m (person class) · SORT tracker (Kalman + Hungarian) · line-crossing counter · Gradio · ZeroGPU

Code: https://github.com/24A31A4235/AI-Based-People-Counting-Analytics-System
