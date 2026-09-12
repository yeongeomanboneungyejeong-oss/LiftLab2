from __future__ import annotations

import gc
import json
import math
import os
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

# CPU-friendly defaults for Render Free / small machines.
ANALYSIS_FPS = float(os.getenv("ANALYSIS_FPS", "8"))
MAX_DIM = int(os.getenv("MAX_DIM", "960"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "416"))
MODEL_NAME = os.getenv("YOLO_MODEL", "yolo11n-pose.pt")

app = FastAPI(title="LiftLab 2", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
model: YOLO | None = None
model_lock = threading.Lock()

# COCO 17-keypoint names used by Ultralytics pose models.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Edges are grouped by body segment so the frontend can render them with colors.
SEGMENTS = {
    "head": {
        "label": "머리",
        "color": "#F59E0B",
        "edges": [[0, 1], [0, 2], [1, 3], [2, 4]],
    },
    "left_arm": {
        "label": "왼팔",
        "color": "#22C55E",
        "edges": [[5, 7], [7, 9]],
    },
    "right_arm": {
        "label": "오른팔",
        "color": "#06B6D4",
        "edges": [[6, 8], [8, 10]],
    },
    "torso": {
        "label": "몸통",
        "color": "#A855F7",
        "edges": [[5, 6], [5, 11], [6, 12], [11, 12]],
    },
    "left_leg": {
        "label": "왼다리",
        "color": "#3B82F6",
        "edges": [[11, 13], [13, 15]],
    },
    "right_leg": {
        "label": "오른다리",
        "color": "#EF4444",
        "edges": [[12, 14], [14, 16]],
    },
}

ANGLE_DEFS = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_shoulder": (7, 5, 11),
    "right_shoulder": (8, 6, 12),
    "left_hip": (5, 11, 13),
    "right_hip": (6, 12, 14),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
}


def set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)


def get_job(job_id: str) -> dict[str, Any] | None:
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def get_model() -> YOLO:
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = YOLO(MODEL_NAME)
    return model


def resize_for_inference(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    if scale == 1.0:
        return frame, 1.0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), scale


def point_angle(a: list[float] | None, b: list[float] | None, c: list[float] | None) -> float | None:
    if not a or not b or not c:
        return None
    va = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    vc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nc = float(np.linalg.norm(vc))
    if na < 1e-6 or nc < 1e-6:
        return None
    cosv = float(np.dot(va, vc) / (na * nc))
    cosv = max(-1.0, min(1.0, cosv))
    return round(math.degrees(math.acos(cosv)), 1)


def calculate_angles(joints: list[list[float] | None]) -> dict[str, float | None]:
    return {
        name: point_angle(joints[a], joints[b], joints[c])
        for name, (a, b, c) in ANGLE_DEFS.items()
    }


def extract_pose(result: Any, width: int, height: int, scale: float) -> list[list[float] | None]:
    if result.keypoints is None or len(result.keypoints) == 0:
        return [None] * 17

    # Largest detected person is the primary subject.
    boxes = result.boxes
    if boxes is not None and len(boxes) > 1:
        areas = boxes.xyxy[:, 0:4]
        area_values = (areas[:, 2] - areas[:, 0]) * (areas[:, 3] - areas[:, 1])
        person_index = int(torch_argmax(area_values))
    else:
        person_index = 0

    xy = result.keypoints.xy[person_index].cpu().numpy()
    conf = None
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[person_index].cpu().numpy()

    joints: list[list[float] | None] = []
    for i in range(17):
        x = float(xy[i][0]) / scale
        y = float(xy[i][1]) / scale
        if conf is not None and float(conf[i]) < 0.25:
            joints.append(None)
        else:
            joints.append([round(max(0.0, min(width, x)), 2), round(max(0.0, min(height, y)), 2)])
    return joints


def torch_argmax(values: Any) -> int:
    # Avoid importing torch at module startup; Ultralytics already depends on it.
    import torch
    return int(torch.argmax(values).item())


def analyze_video(job_id: str, video_path: Path) -> None:
    cap = None
    try:
        set_job(job_id, status="analyzing", progress=0, message="영상 정보를 읽는 중…")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("영상을 열 수 없습니다.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("영상 메타데이터를 읽을 수 없습니다.")

        duration = frame_count / fps
        step = max(1, int(round(fps / ANALYSIS_FPS)))
        sample_indices = list(range(0, frame_count, step))
        if sample_indices[-1] != frame_count - 1:
            sample_indices.append(frame_count - 1)

        set_job(
            job_id,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
            analysis_fps=round(fps / step, 2),
            sampling_step=step,
            message=f"AI 분석 준비 완료 · 약 {len(sample_indices)}개 프레임",
        )

        detector = get_model()
        sampled_frames: list[dict[str, Any]] = []
        sample_set = set(sample_indices)
        next_sample_pos = 0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index in sample_set:
                small_frame, scale = resize_for_inference(frame)
                results = detector.predict(
                    source=small_frame,
                    imgsz=YOLO_IMGSZ,
                    conf=0.25,
                    verbose=False,
                    device="cpu",
                    max_det=3,
                )
                result = results[0]
                joints = extract_pose(result, width, height, scale)
                angles = calculate_angles(joints)
                sampled_frames.append({
                    "frame": frame_index,
                    "time": round(frame_index / fps, 4),
                    "joints": joints,
                    "angles": angles,
                })

                next_sample_pos += 1
                progress = int(next_sample_pos / len(sample_indices) * 100)
                set_job(
                    job_id,
                    progress=progress,
                    message=f"프레임 분석 중 · {next_sample_pos}/{len(sample_indices)}",
                )
                del results, small_frame
                if next_sample_pos % 8 == 0:
                    gc.collect()

            frame_index += 1

        if not sampled_frames:
            raise RuntimeError("사람 포즈를 분석할 프레임이 없습니다.")

        result_payload = {
            "version": "2.0.0",
            "model": MODEL_NAME,
            "fps": round(fps, 4),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": round(duration, 4),
            "analysis_fps": round(fps / step, 2),
            "sampling_step": step,
            "barbell_tracking": False,
            "segments": SEGMENTS,
            "keypoint_names": KEYPOINT_NAMES,
            "angle_names": list(ANGLE_DEFS.keys()),
            "frames": sampled_frames,
        }
        result_file = WORK_DIR / f"{job_id}.json"
        result_file.write_text(json.dumps(result_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

        set_job(job_id, status="completed", progress=100, message="분석 완료", result_path=str(result_file))

    except Exception as exc:
        traceback.print_exc()
        set_job(job_id, status="error", progress=0, message=f"분석 실패: {exc}")
    finally:
        if cap is not None:
            cap.release()
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "LiftLab 2"}


@app.post("/api/analyze")
async def start_analysis(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="영상 파일을 선택해 주세요.")

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    video_path = WORK_DIR / f"{job_id}{suffix}"

    try:
        with video_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"업로드 실패: {exc}") from exc
    finally:
        await file.close()

    # Queue state MUST be written before starting the background thread.
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "분석 대기 중…"}
    threading.Thread(target=analyze_video, args=(job_id, video_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="분석 작업을 찾을 수 없습니다.")
    return job


@app.get("/api/result/{job_id}")
def result(job_id: str) -> FileResponse:
    job = get_job(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="완료된 분석 결과가 없습니다.")
    result_path = Path(str(job["result_path"]))
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="분석 결과 파일이 없습니다.")
    return FileResponse(result_path, media_type="application/json", filename=f"liftlab2_{job_id}.json")
