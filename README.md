# LiftLab 2

모바일 브라우저에서 운동 영상을 업로드하고 YOLO Pose 기반으로 신체 분절과 관절각을 분석하는 FastAPI 웹앱입니다.

## 주요 기능

- 모바일 우선 UI
- 영상 업로드 → 백그라운드 AI 분석
- YOLO11n Pose 기반 17개 COCO keypoint 추적
- 머리 / 왼팔 / 오른팔 / 몸통 / 왼다리 / 오른다리 색상 분리
- 현재 영상 위치와 skeleton/관절각 동기화
- 분석 프레임 기반 움직임 trail
- 관절각 변화 그래프
- JSON / CSV 다운로드
- 바벨 추적 및 바벨 그래프 없음
- CPU 환경을 고려한 프레임 샘플링, 해상도 축소, 416px inference

## 실행

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

브라우저에서 `http://localhost:8000` 접속.

## Render 배포

- Runtime: Docker
- Root Directory: 비워둠
- Build Command: Dockerfile 자동 사용
- Start Command: Dockerfile 자동 사용
- 포트: 7860

GitHub 저장소 전체를 그대로 Render Web Service에 연결하면 됩니다.

## 환경변수

기본값으로도 실행됩니다.

- `ANALYSIS_FPS=8`
- `MAX_DIM=960`
- `YOLO_IMGSZ=416`
- `YOLO_MODEL=yolo11n-pose.pt`
