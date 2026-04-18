import argparse
from pathlib import Path
import cv2
import numpy as np

def color_from_id(track_id: int):
    # deterministic color per ID
    rng = np.random.RandomState(track_id * 9973)
    return int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255))

def load_tracks_mot(txt_path: Path):
    """
    MOT format rows:
    frame, id, x, y, w, h, conf, -1, -1, -1
    """
    data = np.loadtxt(str(txt_path), delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    tracks_by_frame = {}
    for row in data:
        frame = int(row[0])
        tid = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        tracks_by_frame.setdefault(frame, []).append((tid, x, y, w, h))
    return tracks_by_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to sequence folder (contains img1)")
    ap.add_argument("--tracks", required=True, help="Path to MOT track txt")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    seq_path = Path(args.seq)
    img_dir = seq_path / "img1"
    assert img_dir.exists(), f"img1 not found: {img_dir}"

    frames = sorted(img_dir.glob("*.jpg"))
    assert len(frames) > 0, f"No jpg frames in {img_dir}"

    tracks_by_frame = load_tracks_mot(Path(args.tracks))

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    for idx, img_path in enumerate(frames, start=1):
        img = cv2.imread(str(img_path))

        for (tid, x, y, bw, bh) in tracks_by_frame.get(idx, []):
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            c = color_from_id(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img, f"ID:{tid}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        vw.write(img)

    vw.release()
    print(f"[OK] Saved video: {out_path}")

if __name__ == "__main__":
    main()
