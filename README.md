---
title: RCNN Family Comparator
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# RCNN Family Comparator

Web platform for side-by-side comparison of R-CNN, Fast R-CNN, and Faster R-CNN
on user-supplied images under identical conditions.

## Local setup

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py