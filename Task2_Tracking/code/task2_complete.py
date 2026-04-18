# task2_complete.py
# ------------------------------------------------------------
# Task 2: MOT17 Tracking (YOLOv8 detector + simple IoU tracker)
# Outputs:
#   task2_tracking/results/tracks/*.txt   (MOT format)
#   task2_tracking/results/videos/*.mp4   (optional)
#   task2_tracking/results/metrics/tracking_metrics.csv
# ------------------------------------------------------------

import os
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import motmetrics as mm

# ---- NumPy 2.x compatibility patch for motmetrics ----
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a: np.asarray(a, dtype=float)
# -----------------------------------------------------


# =========================
# Utils
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def xyxy_to_xywh(x1, y1, x2, y2):
    return x1, y1, (x2 - x1), (y2 - y1)


def compute_iou_xyxy(a, b):
    # a, b are [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# =========================
# Simple Tracker (IoU based)
# =========================
class Track:
    def __init__(self, det_xyxy_conf, track_id: int):
        self.id = track_id
        self.bbox = det_xyxy_conf[:4].astype(float)
        self.conf = float(det_xyxy_conf[4])
        self.history = deque(maxlen=30)
        self.history.append(self.bbox.copy())
        self.time_since_update = 0

    def predict(self):
        # no motion model (keeps last bbox)
        self.time_since_update += 1
        return self.bbox

    def update(self, det_xyxy_conf):
        self.bbox = det_xyxy_conf[:4].astype(float)
        self.conf = float(det_xyxy_conf[4])
        self.history.append(self.bbox.copy())
        self.time_since_update = 0


class IoUTracker:
    def __init__(self, iou_thr=0.3, max_age=30):
        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.tracks = []
        self.next_id = 1

    def update(self, detections_xyxy_conf: np.ndarray):
        # predict
        for t in self.tracks:
            t.predict()

        # match
        matches = []
        unmatched_dets = list(range(len(detections_xyxy_conf)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(detections_xyxy_conf) > 0 and len(self.tracks) > 0:
            iou_mat = np.zeros((len(detections_xyxy_conf), len(self.tracks)), dtype=float)
            for di, det in enumerate(detections_xyxy_conf):
                for ti, trk in enumerate(self.tracks):
                    iou_mat[di, ti] = compute_iou_xyxy(det[:4], trk.bbox)

            while unmatched_dets and unmatched_tracks:
                best_iou = 0.0
                best_pair = None
                for di in unmatched_dets:
                    for ti in unmatched_tracks:
                        if iou_mat[di, ti] > best_iou:
                            best_iou = iou_mat[di, ti]
                            best_pair = (di, ti)

                if best_pair is None or best_iou < self.iou_thr:
                    break

                di, ti = best_pair
                matches.append((di, ti))
                unmatched_dets.remove(di)
                unmatched_tracks.remove(ti)

        # apply matches
        for di, ti in matches:
            self.tracks[ti].update(detections_xyxy_conf[di])

        # new tracks
        for di in unmatched_dets:
            self.tracks.append(Track(detections_xyxy_conf[di], self.next_id))
            self.next_id += 1

        # remove old
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # return
        return [(t.id, t.bbox.copy(), t.conf) for t in self.tracks]


# =========================
# MOT Tracking Pipeline
# =========================
class MOTPipeline:
    def __init__(self, model_path="yolov8s.pt", conf_thr=0.3, iou_thr=0.3, max_age=30):
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thr = float(conf_thr)
        self.tracker = IoUTracker(iou_thr=iou_thr, max_age=max_age)

    def track_sequence(self, seq_dir: Path, out_txt: Path, save_video=False, out_video: Path | None = None):
        img_dir = seq_dir / "img1"
        images = sorted(img_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No images found in {img_dir}")

        ensure_dir(out_txt.parent)

        vw = None
        if save_video:
            first = cv2.imread(str(images[0]))
            h, w = first.shape[:2]
            ensure_dir(out_video.parent)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(out_video), fourcc, 30, (w, h))

        results = []  # MOT format rows

        for frame_idx, img_path in enumerate(tqdm(images, desc=f"{seq_dir.name}"), start=1):
            frame = cv2.imread(str(img_path))

            # YOLO detect persons only (class 0 = person in COCO)
            pred = self.model(frame, classes=[0], conf=self.conf_thr, verbose=False)[0]

            dets = []
            if pred.boxes is not None and len(pred.boxes) > 0:
                for b in pred.boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu().numpy())
                    dets.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])

            dets = np.array(dets, dtype=float) if dets else np.empty((0, 5), dtype=float)

            tracks = self.tracker.update(dets)

            # save as MOT: frame,id,x,y,w,h,conf,-1,-1,-1
            for tid, bbox, conf in tracks:
                x1, y1, x2, y2 = bbox
                x, y, w, h = xyxy_to_xywh(x1, y1, x2, y2)
                results.append([frame_idx, tid, x, y, w, h, conf, -1, -1, -1])

            # visualize
            if vw is not None:
                for tid, bbox, conf in tracks:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                vw.write(frame)

        if vw is not None:
            vw.release()

        # Write file
        np.savetxt(
            str(out_txt),
            np.array(results, dtype=float),
            fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d"
        )
        print(f"[OK] Saved tracks: {out_txt}")
        if save_video:
            print(f"[OK] Saved video:  {out_video}")


# =========================
# Evaluation
# =========================
def evaluate_mot(results_dir: Path, mot_train_dir: Path, out_csv: Path):
    ensure_dir(out_csv.parent)

    accs = []
    names = []

    for pred_file in sorted(results_dir.glob("*.txt")):
        seq_name = pred_file.stem
        gt_file = mot_train_dir / seq_name / "gt" / "gt.txt"
        if not gt_file.exists():
            print(f"[WARN] Missing GT for {seq_name}: {gt_file}")
            continue

        pred = np.loadtxt(str(pred_file), delimiter=",")
        gt = np.loadtxt(str(gt_file), delimiter=",")

        # MOT17 gt format: frame, id, x, y, w, h, conf, class, vis, ...
        # We use: frame,id,x,y,w,h
        acc = mm.MOTAccumulator(auto_id=True)

        frames = np.unique(gt[:, 0]).astype(int)
        for fr in frames:
            gt_fr = gt[gt[:, 0] == fr]
            pr_fr = pred[pred[:, 0] == fr] if pred.size else np.empty((0, 10))

            gt_ids = gt_fr[:, 1].astype(int)
            gt_boxes = gt_fr[:, 2:6]  # xywh

            if pr_fr.size:
                pr_ids = pr_fr[:, 1].astype(int)
                pr_boxes = pr_fr[:, 2:6]  # xywh
            else:
                pr_ids = np.array([], dtype=int)
                pr_boxes = np.empty((0, 4))

            # iou_matrix expects boxes in (x,y,w,h)
            dists = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
            acc.update(gt_ids, pr_ids, dists)

        accs.append(acc)
        names.append(seq_name)
        print(f"[OK] Evaluated: {seq_name}")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=["mota", "motp", "idf1", "num_switches", "num_frames"],
        names=names
    )

    summary.to_csv(str(out_csv))
    print(f"\n[OK] Saved metrics CSV: {out_csv}")
    print("\n=== SUMMARY ===")
    print(summary)


# =========================
# Main
# =========================
def main():
    project_root = Path(__file__).resolve().parents[2]  # .../cv_competition
    mot_train = project_root / "datasets" / "MOT17" / "train"

    # outputs
    base = project_root / "task2_tracking" / "results"
    tracks_dir = base / "tracks"
    videos_dir = base / "videos"
    metrics_dir = base / "metrics"

    ensure_dir(tracks_dir)
    ensure_dir(videos_dir)
    ensure_dir(metrics_dir)

    if not mot_train.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {mot_train}\n"
            "Expected structure: datasets/MOT17/train/MOT17-xx-YYY/..."
        )

    # Choose only DPM sequences (as in earlier plan)
    sequences = [d for d in mot_train.iterdir() if d.is_dir() and "DPM" in d.name]
    if not sequences:
        # fallback: process all sequences if no DPM
        sequences = [d for d in mot_train.iterdir() if d.is_dir()]

    print(f"[INFO] Found {len(sequences)} sequences under {mot_train}")

    pipeline = MOTPipeline(model_path="yolov8s.pt", conf_thr=0.3, iou_thr=0.3, max_age=30)

    # Track
    for seq in sequences:
        out_txt = tracks_dir / f"{seq.name}.txt"
        out_vid = videos_dir / f"{seq.name}.mp4"
        pipeline.track_sequence(seq, out_txt, save_video=False, out_video=out_vid)

    # Evaluate
    out_csv = metrics_dir / "tracking_metrics.csv"
    evaluate_mot(tracks_dir, mot_train, out_csv)


if __name__ == "__main__":
    main()
