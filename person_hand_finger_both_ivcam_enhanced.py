# -*- coding: utf-8 -*-
"""
=============================================================
  ENHANCED COMPUTER VISION TRACKING SYSTEM  -  v4.0
=============================================================
  MediaPipe Tasks API  (mediapipe 0.10+)
  YOLOv8-Pose for GPU-accelerated arm tracking

  FEATURES
  --------
  [1]  YOLO-POSE person detection + arm keypoints  (GPU)
       Replaces both YOLO person-detect AND MediaPipe pose.
       One model, one GPU call, better accuracy.
  [2]  FULL ARM TRACKING      - shoulders/elbows/wrists + angles
  [3]  HAND TRACKING          - left/right + motion arrow  (CPU)
  [4]  FACE TRACKING          - oval, eyes, iris, gaze, lips,
                                mouth %, head pose  (CPU)
  [5]  EMOTION DETECTION
         Happy / Angry / Neutral
         with confidence bars
  [6]  LIP READING / VISEMES
         Detects mouth shape in real time

  USAGE
  -----
  python person_hand_finger_both_ivcam_enhanced.py ^
      --index 0 --backend dshow --model yolov8s-pose.pt
=============================================================
"""

import time, random, math, argparse, urllib.request, os, collections
import torch
import os

# Force RTX 3050 (GPU 1) - AMD is GPU 0, NVIDIA is GPU 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # RTX 3050 is CUDA device 0

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True   # faster inference
    torch.backends.cudnn.enabled   = True
    _gname = torch.cuda.get_device_name(0)
    print("=" * 50)
    print("  GPU ACTIVE: {}".format(_gname))
    print("  VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory/1e9))
    print("=" * 50)
else:
    DEVICE = "cpu"
    print("WARNING: CUDA not found - running on CPU")
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# COCO keypoint indices for arms (used by YOLOv8-Pose)
#   5=L.Shldr, 6=R.Shldr, 7=L.Elbow, 8=R.Elbow, 9=L.Wrist, 10=R.Wrist
ARM_PAIRS = [(5,7),(7,9),(6,8),(8,10),(5,6)]
ARM_NAMES = {5:"L.Shldr",6:"R.Shldr",7:"L.Elbow",
             8:"R.Elbow",9:"L.Wrist",10:"R.Wrist"}

L_EYE    = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
R_EYE    = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
L_IRIS   = [468,469,470,471,472]
R_IRIS   = [473,474,475,476,477]
LIP_OUT  = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
LIP_IN   = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
FACE_OVAL= [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
            377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

PERSON_COLORS = [
    (255,80,80),(80,255,80),(80,80,255),(255,255,80),
    (255,80,255),(80,255,255),(160,80,255),(255,160,80),
    (80,160,255),(160,255,80),(255,80,160),(80,255,160),
]
FACE_COLORS = [
    (0,230,180),(180,0,230),(230,180,0),(0,180,230),
    (180,230,0),(230,0,180),(120,230,120),(120,120,230),
]

# ── EMOTION COLORS ──────────────────────────────────────────────
EMOTION_COLORS = {
    "Happy":     (0, 220, 100),
    "Angry":     (30,  30, 240),
    "Neutral":   (180, 180, 180),
}
EMOTION_ORDER = ["Happy","Angry","Neutral"]

# Viseme rules: (display_label, description, lambda bs->bool)
# checked top to bottom, first match wins
VISEME_RULES = [
    ("M/B/P", "m b p",   lambda bs: bs("jawOpen")<0.08 and bs("mouthPressLeft")+bs("mouthPressRight")>0.25),
    ("F/V",   "f v",     lambda bs: bs("jawOpen")<0.12 and bs("mouthUpperUpLeft")+bs("mouthUpperUpRight")>0.30),
    ("AA",    "a wide",  lambda bs: bs("jawOpen")>0.55),
    ("AH",    "ah",      lambda bs: bs("jawOpen")>0.40 and bs("mouthSmileLeft")+bs("mouthSmileRight")<0.30),
    ("EE",    "ee",      lambda bs: 0.08<bs("jawOpen")<0.35 and bs("mouthSmileLeft")+bs("mouthSmileRight")>0.35),
    ("OO",    "oo",      lambda bs: bs("jawOpen")<0.30 and bs("mouthPucker")>0.25),
    ("OH",    "oh",      lambda bs: 0.20<bs("jawOpen")<0.50 and bs("mouthFunnel")>0.15),
    ("EH",    "eh",      lambda bs: 0.08<bs("jawOpen")<0.30 and bs("mouthSmileLeft")+bs("mouthSmileRight")<0.25),
    ("TH",    "th",      lambda bs: bs("jawOpen")<0.15 and bs("mouthRollLower")>0.20),
    ("SH",    "sh ch",   lambda bs: bs("jawOpen")<0.15 and bs("mouthShrugUpper")>0.20),
    ("SS",    "s z",     lambda bs: bs("jawOpen")<0.10 and bs("mouthPressLeft")+bs("mouthPressRight")<0.15),
    ("WW",    "w",       lambda bs: bs("mouthPucker")>0.15 and bs("jawOpen")<0.20),
    ("NN",    "n d t",   lambda bs: bs("jawOpen")<0.08),
    ("-",     "silence", lambda bs: True),
]


# ═══════════════════════════════════════════════════════════════
#  TRACKER
# ═══════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self, iou_thr=0.20, max_lost=90, min_area=2500):
        self.nid=1; self.tracks={}
        self.iou_thr=iou_thr; self.max_lost=max_lost; self.min_area=min_area

    @staticmethod
    def iou(a,b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua>0 else 0

    @staticmethod
    def center(b): return ((b[0]+b[2])//2,(b[1]+b[3])//2)

    @staticmethod
    def cdist(a,b):
        ca=Tracker.center(a); cb=Tracker.center(b)
        return math.hypot(ca[0]-cb[0],ca[1]-cb[1])

    @staticmethod
    def make_ip(n):
        random.seed(n*7919)
        return "192.168.{}.{}".format(random.randint(1,254),random.randint(1,254))

    def predict(self,tid):
        t=self.tracks[tid]; b=t['box']
        vx=t.get('vx',0); vy=t.get('vy',0)
        return (b[0]+vx,b[1]+vy,b[2]+vx,b[3]+vy)

    def update(self, raw_dets):
        dets=[d for d in raw_dets if
              max(0,d[2]-d[0])*max(0,d[3]-d[1])>=self.min_area]
        tids=list(self.tracks.keys()); used=set(); matched={}
        for tid in tids:
            pred=self.predict(tid); best,bd=0,-1
            for di,d in enumerate(dets):
                if di in used: continue
                v=self.iou(pred,d)
                if v>best: best,bd=v,di
            if best>=self.iou_thr and bd>=0:
                matched[tid]=bd; used.add(bd)
        diag=math.hypot(1920,1080)*0.25
        for tid in tids:
            if tid in matched: continue
            pred=self.predict(tid); best_d,bd=diag+1,-1
            for di,d in enumerate(dets):
                if di in used: continue
                dist=self.cdist(pred,d)
                if dist<best_d: best_d,bd=dist,di
            if bd>=0 and best_d<=diag:
                matched[tid]=bd; used.add(bd)
        for tid,di in matched.items():
            old=self.tracks[tid]['box']; new=dets[di]
            vx=(new[0]-old[0]+new[2]-old[2])//2
            vy=(new[1]-old[1]+new[3]-old[3])//2
            self.tracks[tid].update({'box':new,'lost':0,'vx':vx,'vy':vy})
        for tid in tids:
            if tid not in matched:
                self.tracks[tid]['lost']+=1
                self.tracks[tid]['vx']=int(self.tracks[tid].get('vx',0)*0.5)
                self.tracks[tid]['vy']=int(self.tracks[tid].get('vy',0)*0.5)
        self.tracks={k:v for k,v in self.tracks.items() if v['lost']<=self.max_lost}
        for di,d in enumerate(dets):
            if di not in used:
                c=PERSON_COLORS[(self.nid-1)%len(PERSON_COLORS)]
                self.tracks[self.nid]={'box':d,'lost':0,'color':c,
                    'ip':self.make_ip(self.nid),'vx':0,'vy':0}
                self.nid+=1
        return [(v['box']+(tid,v['color'],v['ip']))
                for tid,v in self.tracks.items() if v['lost']==0]


# ═══════════════════════════════════════════════════════════════
#  FACE SMOOTHER - adaptive EMA: slow when still, fast when moving
# ═══════════════════════════════════════════════════════════════
class FaceSmoother:
    """
    Adaptive EMA on face landmarks.
    - BASE_ALPHA (0.15): heavy smoothing when face is still → no jitter
    - FAST_ALPHA (0.70): near-raw tracking when face moves fast → no lag
    Nose-tip displacement between frames decides which mode to use.
    Threshold FAST_THRESH (30 px): tune down if still feeling lag.
    """
    BASE_ALPHA  = 0.15    # smoothing coefficient at rest
    FAST_ALPHA  = 0.70    # smoothing coefficient during fast motion
    FAST_THRESH = 30.0    # px displacement of nose tip to enter fast mode
    NOSE_IDX    = 1       # landmark index for nose tip

    def __init__(self, alpha=0.15):
        self.pts = None   # smoothed landmark array (N, 2) float32

    def smooth(self, lms, h, w):
        raw = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        if self.pts is None or self.pts.shape != raw.shape:
            self.pts = raw.copy()
            return self.pts
        # Measure nose-tip displacement to pick smoothing coefficient
        disp = math.hypot(
            float(raw[self.NOSE_IDX, 0] - self.pts[self.NOSE_IDX, 0]),
            float(raw[self.NOSE_IDX, 1] - self.pts[self.NOSE_IDX, 1]))
        a = self.FAST_ALPHA if disp > self.FAST_THRESH else self.BASE_ALPHA
        self.pts = a * raw + (1.0 - a) * self.pts
        return self.pts

# Global smoothers per face slot
_face_smoothers = [FaceSmoother() for _ in range(10)]


# ═══════════════════════════════════════════════════════════════
#  POSE SMOOTHER - EMA for YOLO-Pose GPU keypoints
#
#  Accepts (17,2) pixel coords + (17,) confidence from YOLO-Pose.
#  No Kalman, no velocity, no drift. If YOLO sees the joint →
#  smooth it. If not → hide it immediately.
# ═══════════════════════════════════════════════════════════════
class PoseSmoother:
    ARM_IDXS = [5, 6, 7, 8, 9, 10]   # COCO: shoulders, elbows, wrists
    VIS_THRESH = 0.45                  # keypoint confidence gate
    ALPHA = 0.50                       # EMA: higher=responsive, lower=smooth
    MAX_JUMP = 250                     # reject teleportation glitches

    def __init__(self, **kw):
        self._pts = {}

    def smooth(self, kp_xy, kp_conf, h, w):
        """
        kp_xy:   (17, 2) numpy — pixel coords from YOLO-Pose
        kp_conf: (17,)   numpy — confidence per keypoint
        Returns dict {idx: (px, py)}
        """
        result = {}
        for idx in self.ARM_IDXS:
            conf = float(kp_conf[idx])
            if conf < self.VIS_THRESH:
                self._pts.pop(idx, None)
                continue
            rx, ry = float(kp_xy[idx][0]), float(kp_xy[idx][1])
            if rx < 0 or rx > w or ry < 0 or ry > h:
                self._pts.pop(idx, None)
                continue
            # Jump rejection
            if idx in self._pts:
                ox, oy = self._pts[idx]
                if math.hypot(rx - ox, ry - oy) > self.MAX_JUMP:
                    result[idx] = (int(round(ox)), int(round(oy)))
                    continue
                sx = self.ALPHA * rx + (1.0 - self.ALPHA) * ox
                sy = self.ALPHA * ry + (1.0 - self.ALPHA) * oy
            else:
                sx, sy = rx, ry
            self._pts[idx] = (sx, sy)
            result[idx] = (int(round(sx)), int(round(sy)))
        return result

# Global smoothers per pose slot (up to 10 simultaneous persons)
_pose_smoothers = [PoseSmoother() for _ in range(10)]


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def backend_cv(n):
    n=n.lower()
    if n=="dshow": return cv2.CAP_DSHOW
    if n=="msmf":  return cv2.CAP_MSMF
    return cv2.CAP_ANY

def gpt(lms,i,h,w):
    return (int(lms[i].x*w),int(lms[i].y*h))

def sgpt(spts,i):
    """Get point from smoothed points array."""
    return (int(spts[i][0]), int(spts[i][1]))

def contour(frame,lms,idxs,color,h,w,t=1):
    pts=[gpt(lms,i,h,w) for i in idxs]
    for j in range(len(pts)):
        cv2.line(frame,pts[j],pts[(j+1)%len(pts)],color,t)
    return pts

def angle3(a,b,c):
    ba=np.array([a[0]-b[0],a[1]-b[1]],float)
    bc=np.array([c[0]-b[0],c[1]-b[1]],float)
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-7)
    return math.degrees(math.acos(np.clip(cos,-1,1)))

def bs_get(bs_list,name):
    for b in bs_list:
        if b.category_name==name: return b.score
    return 0.0

def draw_hand(frame,lms,color):
    h,w=frame.shape[:2]; pts=[]
    for lm in lms:
        x,y=int(lm.x*w),int(lm.y*h)
        pts.append((x,y)); cv2.circle(frame,(x,y),3,color,-1)
    for a,b in HAND_CONNECTIONS:
        if a<len(pts) and b<len(pts):
            cv2.line(frame,pts[a],pts[b],color,2)
    return pts

def hand_label(res,i):
    try: return res.handedness[i][0].category_name
    except: return "Hand{}".format(i)

def draw_face_oval(frame,smooth_pts,color,h,w):
    """Draw face oval using pre-smoothed landmark positions."""
    pts=np.array([smooth_pts[i] for i in FACE_OVAL],np.int32)
    ov=frame.copy()
    cv2.fillPoly(ov,[pts],(color[0]//7,color[1]//7,color[2]//7))
    cv2.addWeighted(ov,0.15,frame,0.85,0,frame)
    cv2.polylines(frame,[pts],True,color,2)
    return pts[np.argmin(pts[:,1])]

def draw_iris(frame,lms,idxs,h,w):
    if max(idxs)>=len(lms): return None
    pts=[gpt(lms,i,h,w) for i in idxs]
    cx=int(np.mean([p[0] for p in pts]))
    cy=int(np.mean([p[1] for p in pts]))
    r=max(4,int(np.mean([math.hypot(p[0]-cx,p[1]-cy) for p in pts[1:]]))) if len(pts)>1 else 8
    cv2.circle(frame,(cx,cy),r,(0,210,255),1)
    cv2.circle(frame,(cx,cy),3,(0,0,230),-1)
    return cx,cy

def gaze_dir(eye_idxs,iris_cx,lms,h,w):
    pts=[gpt(lms,i,h,w) for i in eye_idxs]
    lx=min(p[0] for p in pts); rx=max(p[0] for p in pts)
    r=(iris_cx-lx)/(rx-lx+1e-6)
    return "Left" if r<0.37 else ("Right" if r>0.63 else "Center")

def head_pose(spts):
    """
    Improved head pose from smoothed landmarks.
    Uses nose tip, chin, eye corners and ear points.
    """
    def p(i): return spts[i].astype(float)
    nose  = p(1)    # nose tip
    chin  = p(152)  # chin
    le    = p(33)   # right eye outer (camera left)
    re    = p(263)  # left eye outer  (camera right)
    em    = (le+re)/2.0
    # Roll: tilt of eye line
    roll  = math.degrees(math.atan2(float(re[1]-le[1]), float(re[0]-le[0])))
    # Pitch: how far nose is above/below eye midpoint
    face_h = max(1.0, float(np.linalg.norm(chin-em)))
    pitch = ((float(nose[1])-float(em[1]))/face_h - 0.38)*85
    # Yaw: how far nose is left/right of eye midpoint
    face_w = max(1.0, float(np.linalg.norm(re-le)))
    yaw   = ((float(nose[0])-float(em[0]))/face_w)*80
    # Mirror correct yaw (camera flip)
    yaw = -yaw
    return round(yaw,1), round(pitch,1), round(roll,1)

def draw_arms(frame, kp_xy, kp_conf, h, w, smoother):
    """
    Draw arm skeleton for ONE person using YOLO-Pose keypoints.
    kp_xy:   (17, 2) numpy array — pixel coordinates
    kp_conf: (17,)   numpy array — confidence per keypoint
    smoother: PoseSmoother instance for this person slot
    """
    pts = smoother.smooth(kp_xy, kp_conf, h, w)
    if not pts: return
    # Draw skeleton lines — only anatomically plausible ones
    max_limb = int(max(h, w) * 0.40)
    for a, b in ARM_PAIRS:
        if a not in pts or b not in pts: continue
        limb_len = math.hypot(pts[a][0]-pts[b][0], pts[a][1]-pts[b][1])
        if limb_len > max_limb: continue
        if a in (5, 7) or b in (5, 7, 9):     col = (0, 150, 255)   # left arm orange
        elif a in (6, 8) or b in (6, 8, 10):   col = (0, 255, 220)   # right arm teal
        else:                                    col = (220, 220, 220)  # shoulder bridge
        cv2.line(frame, pts[a], pts[b], col, 3)
    # Joint circles + labels
    for idx, pt in pts.items():
        r = 9 if idx in (5, 6) else 6
        cv2.circle(frame, pt, r, (255, 255, 255), -1)
        cv2.circle(frame, pt, r + 2, (100, 100, 100), 1)
        cv2.putText(frame, ARM_NAMES[idx], (pt[0] + 6, pt[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 200), 1)
    # Elbow angles (COCO: 5=L.Shldr, 7=L.Elbow, 9=L.Wrist)
    if all(k in pts for k in (5, 7, 9)):
        a = int(angle3(pts[5], pts[7], pts[9]))
        cv2.putText(frame, "{}d".format(a), (pts[7][0] + 8, pts[7][1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 255), 1)
    if all(k in pts for k in (6, 8, 10)):
        a = int(angle3(pts[6], pts[8], pts[10]))
        cv2.putText(frame, "{}d".format(a), (pts[8][0] + 8, pts[8][1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 220), 1)

# tongue tracking removed

def info_panel(frame,x,y,lines,bg_color=(20,20,20),alpha=0.6):
    try:
        pad=6; lh=18; fw9=9
        pw=max([len(l)*fw9 for l in lines]+[40])+pad*2
        ph=len(lines)*lh+pad*2
        fh,fw=frame.shape[:2]
        x2=min(x+pw,fw-1); y2=min(y+ph,fh-1)
        if x>=x2 or y>=y2: return
        roi=frame[y:y2,x:x2]
        bg=np.full_like(roi,bg_color)
        cv2.addWeighted(bg,alpha,roi,1-alpha,0,roi)
        frame[y:y2,x:x2]=roi
        for i,line in enumerate(lines):
            cv2.putText(frame,line,(x+pad,y+pad+i*lh+lh-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(220,220,220),1)
    except Exception:
        pass

# ── EMOTION SMOOTHER - prevents flickering between emotions ───────
class EmotionSmoother:
    """
    Temporal smoothing for emotion scores.
    Uses exponential moving average + minimum hold time
    so the dominant emotion doesn't flicker every frame.
    """
    def __init__(self, alpha=0.18, hold_frames=6):
        self.alpha  = alpha        # smoothing factor (lower = smoother)
        self.hold   = hold_frames  # frames to hold dominant emotion
        self.smoothed = {}
        self.dominant = "Neutral"
        self.hold_cnt = 0

    def update(self, raw_scores):
        # raw_scores: [(label, score, color), ...]
        raw_dict = {lbl: sc for lbl,sc,_ in raw_scores}
        for lbl in EMOTION_ORDER:
            raw = raw_dict.get(lbl, 0.0)
            prev = self.smoothed.get(lbl, raw)
            self.smoothed[lbl] = self.alpha*raw + (1.0-self.alpha)*prev

        # Dominant: only change if new leader is clearly stronger
        sorted_s = sorted(self.smoothed.items(), key=lambda x:-x[1])
        new_dom = sorted_s[0][0]
        if new_dom != self.dominant:
            # Require new emotion to beat current by margin
            margin = self.smoothed[new_dom] - self.smoothed.get(self.dominant, 0.0)
            if margin > 0.12:
                self.hold_cnt += 1
                if self.hold_cnt >= self.hold:
                    self.dominant = new_dom
                    self.hold_cnt = 0
            else:
                self.hold_cnt = 0
        else:
            self.hold_cnt = 0

        result = [(lbl, self.smoothed[lbl], EMOTION_COLORS[lbl]) for lbl in EMOTION_ORDER]
        result.sort(key=lambda x:-x[1])
        return result, self.dominant

# One smoother per face slot
_emotion_smoothers = [EmotionSmoother(alpha=0.18, hold_frames=6) for _ in range(10)]


def calc_emotions(bs_list):
    """
    Accurate emotion scoring using MediaPipe blendshapes.

    KEY IMPROVEMENTS over v3.0:
    ─────────────────────────────
    1. RESTING-FACE BASELINE: blendshapes are never truly zero even on a
       neutral face. Each signal is now dead-zoned by subtracting a small
       resting threshold before scoring.  This stops "Angry 22%" on a
       perfectly neutral face.
    2. CONFIDENCE GATING: each emotion requires at least one signal above
       a hard gate before it can score.  Random blendshape noise can no
       longer accumulate into a ghost emotion.
    3. LOWER MULTIPLIERS: previous 2.2–5.0× caused saturation; now 1.4–2.5×.
    4. STRONGER NEUTRAL: neutral starts at 1.0 and is only suppressed
       when real emotions are confidently detected.

    Returns sorted list: [(label, score 0-1, color), ...]
    """
    def bs(name): return bs_get(bs_list, name)

    # Helper: subtract resting baseline, clamp to 0
    def dz(val, base=0.06):
        return max(0.0, val - base)

    scores = {}

    # ── HAPPY ──────────────────────────────────────────────────────
    smile  = dz((bs("mouthSmileLeft") + bs("mouthSmileRight")) / 2.0, 0.08)
    cheek  = dz((bs("cheekSquintLeft") + bs("cheekSquintRight")) / 2.0, 0.05)
    frown  = (bs("browDownLeft") + bs("browDownRight")) / 2.0
    # Gate: smile must be clearly above resting
    if smile < 0.05:
        scores["Happy"] = 0.0
    else:
        h = smile * 0.60 + cheek * 0.30 + dz(bs("mouthDimpleLeft"),0.03)*0.05 + dz(bs("mouthDimpleRight"),0.03)*0.05
        h = h * (1.0 - frown * 0.6)
        scores["Happy"] = min(1.0, h * 1.8)

    raw_smile = (bs("mouthSmileLeft") + bs("mouthSmileRight")) / 2.0

    # ── ANGRY ──────────────────────────────────────────────────────
    brow_down  = dz((bs("browDownLeft") + bs("browDownRight")) / 2.0, 0.10)
    nose_sneer = dz((bs("noseSneerLeft") + bs("noseSneerRight")) / 2.0, 0.08)
    jaw_fwd    = dz(bs("jawForward"), 0.03)
    mouth_press= dz((bs("mouthPressLeft") + bs("mouthPressRight")) / 2.0, 0.08)
    # Gate: brows must be clearly pulled down
    if brow_down < 0.04:
        scores["Angry"] = 0.0
    else:
        angry = brow_down*0.45 + nose_sneer*0.25 + mouth_press*0.20 + jaw_fwd*0.10
        angry = angry * (1.0 - raw_smile * 0.9)
        scores["Angry"] = min(1.0, angry * 2.0)

    # ── NEUTRAL ────────────────────────────────────────────────────
    # Neutral should dominate when nothing else is strong.
    # Only suppress it when a real emotion breaks through clearly.
    max_emotion = max(scores.values()) if scores else 0.0
    if max_emotion < 0.10:
        # Nothing remotely detected → strong neutral
        scores["Neutral"] = 1.0
    elif max_emotion < 0.25:
        # Weak signals → neutral still wins
        scores["Neutral"] = max(0.0, 0.80 - max_emotion * 1.5)
    else:
        # Clear emotion → neutral fades
        scores["Neutral"] = max(0.0, 0.50 - max_emotion * 0.8)

    # Build sorted result list
    result = [(lbl, scores[lbl], EMOTION_COLORS[lbl]) for lbl in EMOTION_ORDER]
    result.sort(key=lambda x: -x[1])
    return result

def draw_emotion_panel(frame,x,y,scores,face_color,dominant="Neutral"):
    try:
        fh,fw=frame.shape[:2]
        bar_w=80; pad=4; lh=15; label_w=68
        panel_w=label_w+bar_w+pad*3+28
        panel_h=len(scores)*lh+pad*2+32
        x=max(0,min(fw-panel_w-1,x))
        y=max(0,min(fh-panel_h-1,y))
        ov=frame.copy()
        cv2.rectangle(ov,(x,y),(x+panel_w,y+panel_h),(10,10,10),-1)
        cv2.addWeighted(ov,0.70,frame,0.30,0,frame)
        cv2.rectangle(frame,(x,y),(x+panel_w,y+panel_h),face_color,1)
        cv2.putText(frame,"EMOTION",(x+pad,y+13),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
        # Show stable dominant emotion (from smoother)
        dom_color = EMOTION_COLORS.get(dominant, (180,180,180))
        dom_score = next((s for l,s,_ in scores if l==dominant), 0.0)
        cv2.putText(frame,"{} {:.0f}%".format(dominant, dom_score*100),
                    (x+pad,y+28),cv2.FONT_HERSHEY_SIMPLEX,0.52,dom_color,2)
        for i,(label,score,color) in enumerate(scores):
            ey=y+34+i*lh
            cv2.putText(frame,label,(x+pad,ey+lh-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.30,(180,180,180),1)
            bx=x+pad+label_w
            cv2.rectangle(frame,(bx,ey),(bx+bar_w,ey+lh-3),(40,40,40),-1)
            fill=int(bar_w*score)
            if fill>0:
                cv2.rectangle(frame,(bx,ey),(bx+fill,ey+lh-3),color,-1)
            cv2.putText(frame,"{:.0f}%".format(score*100),
                        (bx+bar_w+3,ey+lh-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,(160,160,160),1)
    except Exception:
        pass

class LipReader:
    def __init__(self,buf_len=10,stable=3):
        self.buf=collections.deque(maxlen=buf_len)
        self.prev=""; self.cnt=0; self.stable=stable

    def detect(self,bs_list):
        def bs(name): return bs_get(bs_list,name)
        for label,desc,cond in VISEME_RULES:
            try:
                if cond(bs): return label,desc
            except Exception:
                pass
        return "-","silence"

    def update(self,bs_list):
        label,desc=self.detect(bs_list)
        if label==self.prev: self.cnt+=1
        else: self.cnt=0; self.prev=label
        if self.cnt==self.stable:
            if not self.buf or self.buf[-1]!=label:
                self.buf.append(label)
        return label,desc

    def sequence(self):
        return " ".join(self.buf)

def draw_lip_panel(frame,x,y,vlabel,vdesc,vseq,face_color):
    try:
        fh,fw=frame.shape[:2]
        panel_w=200; panel_h=54
        x=max(0,min(fw-panel_w-1,x))
        y=max(0,min(fh-panel_h-1,y))
        ov=frame.copy()
        cv2.rectangle(ov,(x,y),(x+panel_w,y+panel_h),(10,10,10),-1)
        cv2.addWeighted(ov,0.70,frame,0.30,0,frame)
        cv2.rectangle(frame,(x,y),(x+panel_w,y+panel_h),face_color,1)
        cv2.putText(frame,"LIP READING",(x+4,y+13),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
        cv2.putText(frame,"Shape:{} ({})".format(vlabel,vdesc),
                    (x+4,y+29),cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,220,255),2)
        cv2.putText(frame,vseq if vseq else "...",(x+4,y+47),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,(160,200,160),1)
    except Exception:
        pass

def download(path,url):
    if not os.path.exists(path):
        print("Downloading {}...".format(os.path.basename(path)))
        urllib.request.urlretrieve(url,path)
        print("  OK")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--index",      type=int,   default=0)
    ap.add_argument("--backend",    type=str,   default="dshow",
                    choices=["dshow","msmf","any"])
    ap.add_argument("--width",      type=int,   default=1920)
    ap.add_argument("--height",     type=int,   default=1080)
    ap.add_argument("--fps",        type=int,   default=60)
    ap.add_argument("--conf",       type=float, default=0.30)
    ap.add_argument("--model",      type=str,   default="yolov8n.pt")
    ap.add_argument("--hand_model", type=str,   default="hand_landmarker.task")
    ap.add_argument("--face_model", type=str,   default="face_landmarker.task")
    ap.add_argument("--no_set_props",action="store_true")
    args=ap.parse_args()

    download(args.face_model,
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

    # ── FIND & OPEN CAMERA ─────────────────────────────────────────
    be = backend_cv(args.backend)

    if args.index >= 0:
        # User specified an index — use it directly
        cap = cv2.VideoCapture(args.index, be)
        if not cap.isOpened():
            print("ERROR: Camera index {} not available!".format(args.index))
            print("  Scanning all cameras...")
            for i in range(10):
                t = cv2.VideoCapture(i, be)
                if t.isOpened():
                    ok, f = t.read()
                    if ok and f is not None and f.mean() > 5:
                        print("  Index {}: WORKS ({}x{})".format(i, f.shape[1], f.shape[0]))
                    t.release()
            return
    else:
        # Auto-scan: find first camera that gives a non-black frame
        print("  Scanning cameras...")
        cap = None
        for i in range(10):
            t = cv2.VideoCapture(i, be)
            if t.isOpened():
                # Read a few frames (some cameras need warmup)
                for _ in range(5): t.read()
                ok, f = t.read()
                if ok and f is not None:
                    brightness = f.mean()
                    if brightness > 5:
                        print("  Index {}: WORKS ({}x{}, brightness={:.0f})".format(
                            i, f.shape[1], f.shape[0], brightness))
                        if cap is None:
                            cap = t
                            args.index = i
                        else:
                            t.release()
                    else:
                        print("  Index {}: black frame (brightness={:.0f})".format(i, brightness))
                        t.release()
                else:
                    print("  Index {}: no frame".format(i))
                    t.release()
            else:
                pass  # not available, skip silently
        if cap is None:
            print("\nERROR: No working camera found!")
            return
        print("  >> Using index {}".format(args.index))

    if not args.no_set_props:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS,          args.fps)

    for _ in range(10): cap.read()
    aw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af=int(cap.get(cv2.CAP_PROP_FPS))
    print("Camera OK  {}x{}  {}fps  index={}".format(aw,ah,af,args.index))

    yolo=YOLO(args.model)
    yolo.to(DEVICE)
    yolo.model.eval()
    if DEVICE=='cuda':
        print("YOLO running on: {}".format(torch.cuda.get_device_name(0)))
    tracker=Tracker()

    hand_lmk=vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.hand_model, delegate=python.BaseOptions.Delegate.GPU if False else python.BaseOptions.Delegate.CPU),
            num_hands=10,running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))

    face_lmk=vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.face_model, delegate=python.BaseOptions.Delegate.GPU if False else python.BaseOptions.Delegate.CPU),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=10,running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35))

    # No separate pose model needed — YOLO-Pose handles it on GPU!
    print("  YOLO-Pose: person detection + arm keypoints (GPU)")
    print("  MediaPipe: face + hands (CPU)")

    lip_readers=[LipReader() for _ in range(10)]
    prev_tip={}; t_prev=time.time()
    fps_buf=collections.deque(maxlen=12)
    _last_tracked = []                # cached tracking results
    _last_kp_map  = {}                # cached {det_idx: (kp_xy, kp_conf)}
    _last_dets    = []                # cached raw detections
    print("All models loaded.  Press q to quit.")

    while True:
        ok,frame=cap.read()
        if not ok or frame is None:
            print("Frame read failed"); break

        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        ts=int(time.time()*1000)

        # 1. YOLO-POSE — person detection + arm keypoints in ONE GPU call
        #    Works with both yolov8s-pose.pt (keypoints) and yolov8n.pt (boxes only)
        try:
            yres=yolo.predict(frame,conf=args.conf,classes=[0],verbose=False,
                              device=DEVICE,max_det=20)
        except Exception as e:
            print("YOLO predict error:", e)
            continue
        dets=[]
        kp_map = {}   # {det_index: (kp_xy, kp_conf)}
        if len(yres)>0 and yres[0].boxes is not None:
            for di, box in enumerate(yres[0].boxes):
                x1,y1,x2,y2=box.xyxy[0].cpu().numpy().astype(int)
                dets.append((x1,y1,x2,y2))
            # Extract keypoints if pose model (graceful fallback if not)
            try:
                if yres[0].keypoints is not None and yres[0].keypoints.xy is not None:
                    kp_all_xy   = yres[0].keypoints.xy.cpu().numpy()    # (N, 17, 2)
                    kp_all_conf = yres[0].keypoints.conf.cpu().numpy()  # (N, 17)
                    for di in range(min(len(dets), len(kp_all_xy))):
                        kp_map[di] = (kp_all_xy[di], kp_all_conf[di])
            except (AttributeError, TypeError):
                pass   # Not a pose model — arms won't show, but no crash
        _last_dets = dets
        _last_kp_map = kp_map
        tracked=tracker.update(dets)
        _last_tracked = tracked

        for (x1,y1,x2,y2,tid,col,ip) in tracked:
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            info_panel(frame,x1,max(0,y1-30),
                       ["ID:{}  IP:{}".format(tid,ip)],
                       bg_color=col,alpha=0.55)

        # 2. ARMS — match tracked persons to YOLO keypoints by IoU
        for (x1,y1,x2,y2,tid,col,ip) in tracked:
            best_iou = 0.3   # minimum IoU to match
            best_di  = -1
            for di, d in enumerate(dets):
                v = Tracker.iou((x1,y1,x2,y2), d)
                if v > best_iou:
                    best_iou = v; best_di = di
            if best_di >= 0 and best_di in kp_map:
                kp_xy, kp_conf = kp_map[best_di]
                si = (tid - 1) % len(_pose_smoothers)
                draw_arms(frame, kp_xy, kp_conf, h, w, _pose_smoothers[si])

        # 3. HANDS
        hand_res=hand_lmk.detect_for_video(mp_img,ts)
        if hand_res.hand_landmarks:
            for i,lms in enumerate(hand_res.hand_landmarks):
                lbl=hand_label(hand_res,i)
                hcol=(60,255,60) if lbl=="Left" else (60,255,255)
                pts=draw_hand(frame,lms,hcol)
                if len(pts)>=9:
                    tip=pts[8]
                    cv2.circle(frame,tip,9,(0,0,255),-1)
                    cv2.circle(frame,tip,11,(255,255,255),1)
                    cv2.putText(frame,"{} Hand".format(lbl),(tip[0]+12,tip[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                    prev=prev_tip.get(lbl)
                    if prev:
                        cv2.arrowedLine(frame,prev,tip,(0,0,255),2,tipLength=0.3)
                        dx2=tip[0]-prev[0]; dy2=tip[1]-prev[1]
                        cv2.putText(frame,"dx={} dy={}".format(dx2,dy2),
                                    (tip[0]+12,tip[1]+12),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,120,255),1)
                    prev_tip[lbl]=tip

        # 4. FACE
        face_res=face_lmk.detect_for_video(mp_img,ts)
        if face_res.face_landmarks:
            for fi,flms in enumerate(face_res.face_landmarks):
                nlms=len(flms)
                fc=FACE_COLORS[fi%len(FACE_COLORS)]

                # Smooth landmarks - kills shaking/jumping
                spts = _face_smoothers[fi].smooth(flms, h, w)

                # Face oval (smooth)
                top_pt = draw_face_oval(frame, spts, fc, h, w)
                cv2.putText(frame, "Face {}".format(fi+1),
                            (int(top_pt[0])-25, int(top_pt[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc, 1)

                # Eye contours (smooth)
                eye_l = [sgpt(spts,i) for i in L_EYE]
                eye_r = [sgpt(spts,i) for i in R_EYE]
                for j in range(len(eye_l)):
                    cv2.line(frame, eye_l[j], eye_l[(j+1)%len(eye_l)], (0,240,240), 1)
                for j in range(len(eye_r)):
                    cv2.line(frame, eye_r[j], eye_r[(j+1)%len(eye_r)], (0,240,240), 1)

                # Iris + gaze (L/R swapped for mirror)
                l_gaze = r_gaze = "--"
                if nlms > 477:
                    ri = draw_iris(frame, flms, L_IRIS, h, w)
                    li = draw_iris(frame, flms, R_IRIS, h, w)
                    if li:
                        raw = gaze_dir(R_EYE, li[0], flms, h, w)
                        l_gaze = "Right" if raw=="Left" else ("Left" if raw=="Right" else raw)
                        cv2.putText(frame, "L:{}".format(l_gaze), (li[0]-22, li[1]-16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,210,255), 1)
                    if ri:
                        raw = gaze_dir(L_EYE, ri[0], flms, h, w)
                        r_gaze = "Right" if raw=="Left" else ("Left" if raw=="Right" else raw)
                        cv2.putText(frame, "R:{}".format(r_gaze), (ri[0]-22, ri[1]-16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,210,255), 1)

                # Lip contours (smooth)
                lip_o = [sgpt(spts,i) for i in LIP_OUT]
                lip_i = [sgpt(spts,i) for i in LIP_IN]
                for j in range(len(lip_o)):
                    cv2.line(frame, lip_o[j], lip_o[(j+1)%len(lip_o)], (0,170,255), 1)
                for j in range(len(lip_i)):
                    cv2.line(frame, lip_i[j], lip_i[(j+1)%len(lip_i)], (0,130,200), 1)

                # Blendshapes
                bs_list = []; jaw_bs = 0.0
                if face_res.face_blendshapes and fi < len(face_res.face_blendshapes):
                    bs_list = face_res.face_blendshapes[fi]
                    jaw_bs  = bs_get(bs_list, "jawOpen")

                # Mouth open % - use BOTH landmark distance and blendshape
                # Landmark method: distance between upper/lower inner lip
                upper_lip_y = (flms[13].y + flms[312].y + flms[311].y) / 3.0
                lower_lip_y = (flms[14].y + flms[317].y + flms[318].y) / 3.0
                mouth_h_px  = abs(lower_lip_y - upper_lip_y) * h
                face_h_px   = abs(flms[10].y  - flms[152].y) * h + 1e-6
                # Normalize by face height so it works at any distance
                mouth_ratio = mouth_h_px / face_h_px
                # At rest ~0.01, slightly open ~0.03, fully open ~0.12+
                jaw_lm = min(1.0, mouth_ratio / 0.10)
                # Blend both methods - landmark is more reliable
                jaw = jaw_lm * 0.65 + jaw_bs * 0.35
                mouth_pct   = min(100, int(jaw * 100))
                mouth_state = "Open {}%".format(mouth_pct) if mouth_pct > 5 else "Closed"

                # Head pose (smooth)
                yaw2, pitch2, roll2 = head_pose(spts)

                # Emotions - use smoother to prevent flickering
                # Always pass jaw to emotion even if bs_list partial
                if bs_list and len(bs_list) > 10:
                    raw_scores = calc_emotions(bs_list)
                    emotion_scores, dom_emotion = _emotion_smoothers[fi].update(raw_scores)
                else:
                    # No blendshapes - neutral
                    neutral_raw = [("Neutral",1.0,(180,180,180))] +                                   [(lbl,0.0,EMOTION_COLORS[lbl]) for lbl in EMOTION_ORDER if lbl!="Neutral"]
                    emotion_scores, dom_emotion = _emotion_smoothers[fi].update(neutral_raw)

                # Lip reading
                vlabel = vdesc = "-"; vseq = ""
                if bs_list and fi < len(lip_readers):
                    vlabel, vdesc = lip_readers[fi].update(bs_list)
                    vseq = lip_readers[fi].sequence()

                # Info panel pinned to right edge of face oval
                face_right = int(np.max(spts[FACE_OVAL, 0]))
                nose_y     = int(spts[1][1])
                px = min(face_right + 8, w - 165)
                py = max(4, nose_y - 35)

                info_panel(frame, px, py, [
                    "Face {}".format(fi+1),
                    "Mouth: {}".format(mouth_state),
                    "Gaze  L:{} R:{}".format(l_gaze, r_gaze),
                    "Yaw:  {}deg".format(yaw2),
                    "Pitch:{}deg".format(pitch2),
                    "Roll: {}deg".format(roll2),
                ], bg_color=(10,10,10), alpha=0.68)

                if emotion_scores:
                    draw_emotion_panel(frame, px, py+112, emotion_scores, fc, dom_emotion)

                ep_h = len(emotion_scores)*15+34+6 if emotion_scores else 0
                draw_lip_panel(frame, px, py+112+ep_h+4, vlabel, vdesc, vseq, fc)

        # 5. HUD
        now=time.time()
        fps_buf.append(1.0/max(1e-6,now-t_prev)); t_prev=now
        fps_val=sum(fps_buf)/len(fps_buf)
        faces_n=len(face_res.face_landmarks) if face_res.face_landmarks else 0
        hands_n=len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
        info_panel(frame,10,10,[
            "FPS:     {:.1f}".format(fps_val),
            "Persons: {}".format(len(tracked)),
            "Faces:   {}".format(faces_n),
            "Hands:   {}".format(hands_n),
        ],bg_color=(0,0,0),alpha=0.65)

        # Large FPS overlay — top-right corner, always visible
        fps_txt = "{:.0f} fps".format(fps_val)
        (tw, th), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, fps_txt, (w - tw - 12, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, fps_txt, (w - tw - 12, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 100), 2)

        cv2.imshow("Enhanced Tracking  [q=quit]",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_lmk.close(); face_lmk.close()
    print("Done.")


if __name__=="__main__":
    main()
