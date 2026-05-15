"""
SUTERUP V5.1 — Surgical Suture Training Platform
Programmer: Guneet Singh

INSTALL:
    pip install streamlit opencv-python numpy pillow pandas

RUN:
    streamlit run suterup_v5.py

CAMERA MODES (Live Training page):
    Snapshot Mode  — take one picture, get instant analysis
    Live Mode      — continuous auto-refresh loop, real-time feedback

PAGES:
    Live Training   — camera + analysis
    Session History — run summaries, score chart, event log
    Calibration     — manual HSV sliders + mask preview
    About           — scoring rubric, tech stack
"""

import streamlit as st
import cv2
import numpy as np
import math
import json
import time
from datetime import datetime
from collections import deque
from PIL import Image
import io

# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suterup",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
#  CSS — clean dark surgical UI, no gradients, no flash
# ─────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
    background: #0d1117 !important;
    color: #cbd5e1 !important;
    font-family: 'DM Mono', monospace !important;
}

/* Prevent any flash / white blink on load */
.stApp { animation: none !important; transition: none !important; }

/* ── Main content padding ── */
.block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; max-width: 1300px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0f14 !important;
    border-right: 1px solid #1e2d3d !important;
}
[data-testid="stSidebar"] * { color: #64829a !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.82em !important; }

/* ── Sidebar brand ── */
.brand-title {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    font-size: 1.3em;
    color: #e2e8f0 !important;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 0.62em;
    color: #2a4a5e !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Buttons — two variants ── */
.stButton > button {
    background: #111927 !important;
    border: 1px solid #1e3a4f !important;
    color: #94b4c8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76em !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    letter-spacing: 0.06em !important;
    padding: 9px 18px !important;
    transition: border-color 0.12s, color 0.12s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: #4a8aaa !important;
    color: #c9dce8 !important;
    background: #131e2a !important;
}
.stButton > button:disabled {
    border-color: #111820 !important;
    color: #1e3040 !important;
}
/* Active/recording state — applied via container div */
.btn-active > .stButton > button {
    border-color: #c0392b !important;
    color: #e87070 !important;
}
.btn-primary > .stButton > button {
    border-color: #2e6e8a !important;
    color: #7ec8e3 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0f1923 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #3a607a !important;
    font-size: 0.68em !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #c8dce8 !important;
    font-size: 1.35em !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.7em !important; }

/* ── Sliders ── */
.stSlider > div { padding: 0 !important; }
.stSlider label { color: #3a6070 !important; font-size: 0.72em !important; }
input[type=range] { accent-color: #4a8aaa !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #0f1923 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    color: #4a7a8a !important;
    font-size: 0.78em !important;
}

/* ── Alerts (st.success / st.error / st.warning) ── */
.stAlert {
    background: #0f1923 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 4px !important;
    font-size: 0.78em !important;
}
.stAlert p { color: #7aaac8 !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #1a2a38 !important; margin: 16px 0 !important; }

/* ── Camera widget ── */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border: 1px solid #1e2d3d !important;
    border-radius: 4px !important;
}
[data-testid="stCameraInput"] label { display: none !important; }
[data-testid="stCameraInput"] button {
    background: #0f1923 !important;
    border: 1px solid #1e3a4f !important;
    color: #7aaac8 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75em !important;
}

/* ── Annotated image output ── */
[data-testid="stImage"] img {
    border: 1px solid #1e2d3d !important;
    border-radius: 4px !important;
    display: block !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.76em !important;
    color: #3a6070 !important;
    letter-spacing: 0.06em !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 14px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #94b4c8 !important;
    border-bottom: 1px solid #4a8aaa !important;
}

/* ── Custom HTML components ── */
.page-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.25em;
    font-weight: 600;
    color: #c8dce8;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}
.page-sub {
    font-size: 0.68em;
    color: #243545;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 22px;
}
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65em;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #243545;
    margin-bottom: 8px;
    margin-top: 4px;
}
.card {
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-radius: 4px;
    padding: 14px 16px;
    font-size: 0.8em;
    line-height: 1.85;
}
.mode-badge {
    display: inline-block;
    font-size: 0.65em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 2px;
    margin-left: 8px;
    vertical-align: middle;
}
.mode-snapshot { background: #0f1d28; color: #4a8aaa; border: 1px solid #1e3a50; }
.mode-live     { background: #200f0f; color: #aa5a5a; border: 1px solid #4a1e1e; }
.score-bar-bg {
    background: #0a1018;
    border-radius: 2px;
    height: 4px;
    width: 100%;
    margin-top: 6px;
    margin-bottom: 10px;
}
.score-bar-fill { height: 4px; border-radius: 2px; }
.log-box {
    max-height: 300px;
    overflow-y: auto;
    background: #080c10;
    border: 1px solid #1a2838;
    border-radius: 4px;
    padding: 10px 12px;
}
.log-line {
    font-size: 0.72em;
    color: #3a6070;
    border-bottom: 1px solid #0e1820;
    padding: 3px 0;
    line-height: 1.6;
}
.grade-optimal { color: #4db88a !important; }
.grade-warn    { color: #c8a53a !important; }
.grade-danger  { color: #b85a5a !important; }
.status-indicator {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}
.status-live    { background: #b85a5a; box-shadow: 0 0 4px #b85a5a88; }
.status-standby { background: #2a4050; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────
def _init():
    defaults = {
        "hsv":              dict(h_min=90, h_max=130, s_min=80, s_max=255, v_min=50, v_max=255),
        "area":             dict(min_area=200, max_area=8000),
        "hsv_source":       "default",
        "run_active":       False,
        "run_stitches":     [],
        "session_total":    0,
        "log_lines":        [],
        "run_history":      [],
        "centroid_history": deque(maxlen=6),
        "auto_hsv_result":  None,
        # Camera mode: "snapshot" | "live"
        "cam_mode":         "snapshot",
        # Flag: live mode is actively looping
        "live_running":     False,
        # Cached last processed frame (BGR numpy)
        "last_annotated":   None,
        # Cached last centroids for log button
        "last_centroids":   [],
        "page":             "Live Training",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────
#  MATH + GRADING
# ─────────────────────────────────────────────────────
def seg_dist(p1, p2):
    """Euclidean pixel distance between two (x,y) points."""
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def precision_score(d):
    """
    Linear score 0-100 with peak at 90px.
    Score = 100 - |d - 90| / 0.9
    Rationale: 90px is the optimal inter-dot spacing at ~30cm camera distance.
    Each pixel of deviation costs 1/0.9 ≈ 1.1 points; clamped at 0.
    """
    return max(0, int(100 - abs(d - 90) / 0.9))

# (distance_upper_bound, label, bgr_color, feedback)
GRADE_TABLE = [
    (40,  "TOO CLOSE", (100, 80,  200), "Stitches too close — widen placement"),
    (60,  "SHORT",     (60,  160, 210), "Slightly narrow — move dots apart"),
    (120, "OPTIMAL",   (80,  185, 130), "Perfect — within ideal suture range"),
    (160, "WIDE",      (60,  160, 210), "Slightly wide — bring dots closer"),
    (999, "TOO WIDE",  (100, 80,  200), "Too wide — reduce stitch distance"),
]

def grade(d):
    """Return (label, bgr_color, feedback) for a pixel distance."""
    for threshold, label, color, feedback in GRADE_TABLE:
        if d < threshold:
            return label, color, feedback
    return "TOO WIDE", (100, 80, 200), "Too wide — reduce stitch distance"

def bgr_to_hex(bgr):
    """OpenCV BGR tuple → CSS hex (#rrggbb)."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

def grade_class(label):
    """Map grade label → CSS class for coloring HTML text."""
    return {"OPTIMAL": "grade-optimal"}.get(
        label,
        "grade-danger" if label in ("TOO CLOSE", "TOO WIDE") else "grade-warn"
    )

# ─────────────────────────────────────────────────────
#  AUTO-HSV (6-pass engine, unchanged from V5.0)
# ─────────────────────────────────────────────────────
def auto_tune_hsv(frame_bgr):
    """
    6-pass adaptive HSV tuner. Returns parameter dict or None.
    Pass 1: CLAHE normalise lighting.
    Pass 2: Extract candidate pixels by adaptive sat/val floors.
    Pass 3: Hue histogram, suppress background hues >4% of frame.
    Pass 4: Find peak hue and measure cluster spread.
    Pass 5: Validate with circularity blob check; relax if 0 blobs.
    Pass 6: Scale area bounds to frame resolution.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # Pass 1 — CLAHE on Lab L channel for lighting normalisation
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_Lab2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Pass 2 — adaptive candidate floors (22% of mean brightness, 28% of mean sat)
    mean_v, mean_s = float(np.mean(v_ch)), float(np.mean(s_ch))
    v_floor = max(15, int(mean_v * 0.22))
    s_floor = max(25, int(mean_s * 0.28))
    candidate_mask = (s_ch > s_floor) & (v_ch > v_floor)
    if candidate_mask.sum() < 80:
        return None

    candidate_hues = h_ch[candidate_mask]

    # Pass 3 — hue histogram + background suppression
    hist, _ = np.histogram(candidate_hues, bins=180, range=(0, 179))
    hist_sm = np.convolve(hist.astype(float), np.ones(9)/9, mode='same')
    total_px = frame_bgr.shape[0] * frame_bgr.shape[1]
    hist_sm[hist_sm > total_px * 0.04] = 0   # zero-out dominant background hues
    if hist_sm.max() < 30:
        return None

    # Pass 4 — peak hue + spread measurement
    peak_hue = int(np.argmax(hist_sm))
    peak_val = hist_sm[peak_hue]
    thresh20 = peak_val * 0.20
    spread = 0
    for delta in range(1, 35):
        left  = hist_sm[(peak_hue - delta) % 180] > thresh20
        right = hist_sm[(peak_hue + delta) % 180] > thresh20
        if left or right:
            spread = delta
        else:
            break

    tol   = max(14, spread + 6)
    h_min = max(0,   peak_hue - tol)
    h_max = min(179, peak_hue + tol)
    if (h_max - h_min) < 16:
        c = (h_min + h_max) // 2
        h_min, h_max = max(0, c-8), min(179, c+8)

    # Derive s/v bounds from pixels near peak hue (8th percentile as floor)
    near_peak = candidate_mask & (np.abs(h_ch.astype(int) - peak_hue) < tol)
    sat_near  = s_ch[near_peak]
    val_near  = v_ch[near_peak]
    s_min = max(25, int(np.percentile(sat_near, 8))) if len(sat_near) > 0 else s_floor
    v_min = max(15, int(np.percentile(val_near, 8))) if len(val_near) > 0 else v_floor

    # Pass 5 — validate with circularity blob check
    min_area = max(40, int(total_px * 0.00025))
    max_area = min(int(total_px * 0.05), 22000)
    test_mask = _build_mask_internal(enhanced, h_min, h_max, s_min, 255, v_min, 255)
    cnts, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            perim = cv2.arcLength(c, True)
            if perim > 0 and (4 * math.pi * area / perim**2) > 0.25:
                valid_blobs += 1
    if valid_blobs == 0:
        s_min = max(20, s_min - 20)
        v_min = max(15, v_min - 20)

    # Pass 6 — adaptive area from frame height
    fh = frame_bgr.shape[0]
    r_min = max(3, int(fh * 0.008))
    r_max = max(r_min + 5, int(fh * 0.06))
    min_area = max(40, int(math.pi * r_min**2 * 0.7))
    max_area = min(22000, int(math.pi * r_max**2 * 1.3))

    return {
        "h_min": h_min, "h_max": h_max,
        "s_min": s_min, "s_max": 255,
        "v_min": v_min, "v_max": 255,
        "min_area": min_area, "max_area": max_area,
        "peak_hue": peak_hue,
        "valid_blobs": valid_blobs,
        "spread": spread, "tol": tol,
    }

# ─────────────────────────────────────────────────────
#  MASK BUILDING
# ─────────────────────────────────────────────────────
def _build_mask_internal(frame_bgr, h_min, h_max, s_min, s_max, v_min, v_max):
    """
    Build binary HSV mask with CLAHE normalisation and morphological cleanup.
    OPEN removes speckle noise; CLOSE fills holes in blobs.
    Both use elliptical 5x5 kernel (matches circular dot shape).
    """
    h_min, h_max = sorted([int(np.clip(h_min,0,179)), int(np.clip(h_max,0,179))])
    s_min, s_max = sorted([int(np.clip(s_min,0,255)), int(np.clip(s_max,0,255))])
    v_min, v_max = sorted([int(np.clip(v_min,0,255)), int(np.clip(v_max,0,255))])

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_Lab2BGR)

    hsv  = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([h_min, s_min, v_min], dtype=np.uint8),
                       np.array([h_max, s_max, v_max], dtype=np.uint8))

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

# ─────────────────────────────────────────────────────
#  BLOB DETECTION
# ─────────────────────────────────────────────────────
def detect_blobs(frame_bgr):
    """
    Detect suture dot centroids.
    Filters: area bounds + circularity >= 0.20 (rejects non-dot shapes).
    Orders by nearest-neighbour chain (left-to-right physical stitch order).
    Returns (centroids_list, mask).
    """
    h_p = st.session_state.hsv
    a_p = st.session_state.area
    min_a = max(10, int(a_p["min_area"]))
    max_a = max(min_a + 100, int(a_p["max_area"]))

    mask = _build_mask_internal(
        frame_bgr,
        h_p["h_min"], h_p["h_max"],
        h_p["s_min"], h_p["s_max"],
        h_p["v_min"], h_p["v_max"],
    )
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_a <= area <= max_a):
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        # Circularity = 4π·A/P² — 1.0 for perfect circle; dots ≥ 0.20
        if (4 * math.pi * area / perim**2) < 0.20:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            pts.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]), area))

    if len(pts) < 2:
        return [(p[0],p[1]) for p in pts[:1]], mask

    pts = sorted(pts, key=lambda p: p[0])
    ordered = [pts.pop(0)]
    while pts:
        last = ordered[-1]
        idx = min(range(len(pts)), key=lambda i: (pts[i][0]-last[0])**2+(pts[i][1]-last[1])**2)
        ordered.append(pts.pop(idx))

    return [(p[0],p[1]) for p in ordered[:10]], mask

# ─────────────────────────────────────────────────────
#  TEMPORAL SMOOTHING
# ─────────────────────────────────────────────────────
def smooth_centroids(new_centroids):
    """
    Average centroids over last N frames (deque buffer in session state).
    Only smooths if dot count is stable across all buffered frames.
    Averaging with mismatched counts produces garbage — so skip in that case.
    """
    history = st.session_state.centroid_history
    if not new_centroids:
        history.clear()
        return new_centroids
    history.append(new_centroids)
    counts = [len(f) for f in history]
    if len(set(counts)) != 1:
        return new_centroids   # count changed — use raw
    n, k = len(history), len(new_centroids)
    return [
        (int(sum(history[j][i][0] for j in range(n))/n),
         int(sum(history[j][i][1] for j in range(n))/n))
        for i in range(k)
    ]

# ─────────────────────────────────────────────────────
#  FRAME ANNOTATION
# ─────────────────────────────────────────────────────
# Muted dot color palette — no neons
DOT_COLORS_BGR = [
    (120, 190, 110), (100, 160, 210), (150, 120, 200),
    (100, 200, 170), (200, 160,  80), (140, 190, 220),
    ( 80, 140, 200), (190, 130,  90), (110, 200, 140),
    (160, 100, 180),
]

def draw_dashed(frame, p1, p2, color, thickness=2):
    """Dashed line between p1 and p2: 10px on, 5px off segments."""
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    length = max(1, math.sqrt(dx**2+dy**2))
    ux, uy = dx/length, dy/length
    pos, on = 0.0, True
    while pos < length:
        end = min(pos + (10 if on else 5), length)
        if on:
            a = (int(p1[0]+ux*pos), int(p1[1]+uy*pos))
            b = (int(p1[0]+ux*end), int(p1[1]+uy*end))
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
        pos, on = end, not on

def annotate_frame(frame_bgr, centroids):
    """
    Draw overlays on a copy of the frame:
    - Dashed lines between dot pairs with distance + grade labels
    - Dot circles (outer ring + filled inner + bright center)
    - Top-left HUD: dot count, avg score, avg distance, HSV mode
    - Bottom status bar: REC/STANDBY + session stitch count
    Returns annotated BGR frame.
    """
    out = frame_bgr.copy()
    fh, fw = out.shape[:2]

    if len(centroids) >= 2:
        dists  = [seg_dist(centroids[i], centroids[i+1]) for i in range(len(centroids)-1)]
        scores = [precision_score(d) for d in dists]
        avg_sc = int(sum(scores)/len(scores))
        avg_d  = sum(dists)/len(dists)

        for i in range(len(centroids)-1):
            p1, p2 = centroids[i], centroids[i+1]
            d = dists[i]
            lbl, col, _ = grade(d)
            sc = scores[i]
            draw_dashed(out, p1, p2, col, 2)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            for txt, dy_off in [(f"{d:.0f}px", -18), (f"{lbl} {sc}/100", 0)]:
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(out, (mid[0]-3, mid[1]+dy_off-th-2),
                              (mid[0]+tw+3, mid[1]+dy_off+2), (6, 10, 16), -1)
                cv2.putText(out, txt, (mid[0], mid[1]+dy_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

        for i, pt in enumerate(centroids):
            c = DOT_COLORS_BGR[i % len(DOT_COLORS_BGR)]
            cv2.circle(out, pt, 16, c, 1, cv2.LINE_AA)
            cv2.circle(out, pt,  9, c, -1, cv2.LINE_AA)
            cv2.circle(out, pt,  2, (230, 240, 235), -1, cv2.LINE_AA)
            cv2.putText(out, f"P{i+1}", (pt[0]+18, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)

        # HUD top-left
        score_col = (80,185,130) if avg_sc>=70 else (60,160,210) if avg_sc>=40 else (100,80,200)
        hud = [
            (f"DOTS: {len(centroids)}", (140, 180, 160)),
            (f"SCORE: {avg_sc}/100",    score_col),
            (f"DIST: {avg_d:.0f}px",    (140, 180, 160)),
            (f"HSV: {st.session_state.hsv_source.upper()}", (80, 150, 120)),
        ]
        for j, (txt, col) in enumerate(hud):
            y = 10 + j*20 + 16
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(out, (6, y-th-2), (10+tw+2, y+2), (5, 9, 14), -1)
            cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # Bottom status bar
        is_rec = st.session_state.run_active
        rec_txt = "REC" if is_rec else "STANDBY"
        rec_col = (80, 100, 200) if is_rec else (50, 80, 70)
        cv2.rectangle(out, (0, fh-24), (fw, fh), (5, 9, 14), -1)
        cv2.putText(out, f"SUTERUP  {rec_txt}  STITCHES: {st.session_state.session_total}",
                    (8, fh-7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, rec_col, 1, cv2.LINE_AA)

    else:
        msg = "NO DOTS — run Auto-Tune or check Calibration" if not centroids else "1 DOT — need 2+"
        oy = fh // 2
        cv2.rectangle(out, (0, oy-22), (fw, oy+10), (6, 10, 16), -1)
        cv2.putText(out, msg, (10, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 110, 160), 1, cv2.LINE_AA)

    return out

# ─────────────────────────────────────────────────────
#  CAMERA HELPERS
# ─────────────────────────────────────────────────────
def camera_to_bgr(cam_file):
    """
    Streamlit camera_input → OpenCV BGR numpy array.
    Reads JPEG bytes from the file buffer, decodes to BGR via cv2.imdecode.
    Returns None on any failure (no crash).
    """
    if cam_file is None:
        return None
    try:
        raw = cam_file.read()
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr
    except Exception:
        return None

def bgr_to_pil(bgr):
    """BGR numpy → RGB PIL Image for st.image() (Streamlit expects RGB)."""
    if bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def process_frame(cam_file):
    """
    Full pipeline: camera file → annotated PIL image + centroids.
    Returns (pil_image, centroids). Updates session state cache.
    Safe — returns (None, []) on any error.
    """
    bgr = camera_to_bgr(cam_file)
    if bgr is None:
        return None, []
    try:
        centroids_raw, _ = detect_blobs(bgr)
        centroids = smooth_centroids(centroids_raw)
        annotated = annotate_frame(bgr, centroids)
        st.session_state.last_annotated = annotated
        st.session_state.last_centroids = centroids
        return bgr_to_pil(annotated), centroids
    except Exception as e:
        st.session_state.last_annotated = bgr
        st.session_state.last_centroids = []
        return bgr_to_pil(bgr), []

# ─────────────────────────────────────────────────────
#  SCORE CARD HTML  (reused in snapshot + live mode)
# ─────────────────────────────────────────────────────
def render_score_card(centroids):
    """
    Build and render the analysis card below the annotated image.
    Shows grade, score bar, segment table, and summary stats.
    """
    if len(centroids) < 2:
        if len(centroids) == 1:
            st.markdown('<div class="card" style="color:#c8a53a;">⚠ 1 dot found — need at least 2</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="color:#3a6070;">Waiting for dots...</div>',
                        unsafe_allow_html=True)
        return

    dists  = [seg_dist(centroids[i], centroids[i+1]) for i in range(len(centroids)-1)]
    scores = [precision_score(d) for d in dists]
    avg_sc = int(sum(scores)/len(scores))
    avg_d  = sum(dists)/len(dists)
    last_d = dists[-1]
    label, col_bgr, feedback = grade(last_d)
    hex_col = bgr_to_hex(col_bgr)
    gc = grade_class(label)

    bar_color = "#4db88a" if avg_sc>=70 else "#c8a53a" if avg_sc>=40 else "#b85a5a"

    rows = "".join(
        f"<tr>"
        f"<td style='color:#2a4a5a;padding:3px 10px;'>P{i+1}→P{i+2}</td>"
        f"<td class='{grade_class(grade(d)[0])}' style='padding:3px 10px;'>{d:.1f}px</td>"
        f"<td class='{grade_class(grade(d)[0])}' style='padding:3px 10px;'>{grade(d)[0]}</td>"
        f"<td class='{grade_class(grade(d)[0])}' style='padding:3px 10px;'>{sc}/100</td>"
        f"</tr>"
        for i, (d, sc) in enumerate(zip(dists, scores))
    )

    st.markdown(f"""
    <div class="card" style="margin-top:10px;">
      <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px;">
        <span class="{gc}" style="font-family:'DM Sans',sans-serif;font-size:1.05em;font-weight:600;">
          {label}
        </span>
        <span style="color:#2a4a5a;font-size:0.8em;">{avg_sc}/100</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{avg_sc}%;background:{bar_color};"></div>
      </div>
      <div style="color:#2e5060;font-size:0.75em;margin-bottom:10px;">{feedback}</div>
      <table style="width:100%;border-collapse:collapse;font-size:0.74em;">
        <tr style="color:#1a3040;border-bottom:1px solid #0e1820;">
          <td style="padding:3px 10px;">Segment</td>
          <td style="padding:3px 10px;">Distance</td>
          <td style="padding:3px 10px;">Grade</td>
          <td style="padding:3px 10px;">Score</td>
        </tr>
        {rows}
      </table>
      <div style="margin-top:10px;color:#1e3a4a;font-size:0.7em;">
        Avg dist: <span style="color:#4a7a8a;">{avg_d:.1f}px</span>
        &nbsp;·&nbsp; Dots: <span style="color:#4a7a8a;">{len(centroids)}</span>
        &nbsp;·&nbsp; Session: <span style="color:#4a7a8a;">{st.session_state.session_total}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  LOG CURRENT FRAME (shared by both modes)
# ─────────────────────────────────────────────────────
def log_frame():
    """
    Save current centroids as stitch measurements to run log.
    Increments session_total and appends structured log lines.
    """
    cents = st.session_state.last_centroids
    if len(cents) < 2:
        st.session_state.log_lines.append("⚠ Log skipped — fewer than 2 dots visible")
        return 0
    ts = datetime.now().strftime("%H:%M:%S")
    count = 0
    for i in range(len(cents)-1):
        d  = seg_dist(cents[i], cents[i+1])
        sc = precision_score(d)
        gl, _, _ = grade(d)
        st.session_state.run_stitches.append({"d": d, "sc": sc})
        st.session_state.session_total += 1
        count += 1
        st.session_state.log_lines.append(
            f"[{ts}] #{st.session_state.session_total} P{i+1}→P{i+2} "
            f"| {d:.1f}px | {gl} | {sc}/100"
        )
    st.session_state.log_lines.append(f"+ {count} stitch(es) saved")
    return count

# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 16px 0;">
      <div class="brand-title">⚕ Suterup</div>
      <div class="brand-sub">Surgical Training · v5.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #1a2838;margin-bottom:12px;"></div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Live Training", "Session History", "Calibration", "About"],
        index=["Live Training","Session History","Calibration","About"].index(
            st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown('<div style="border-top:1px solid #1a2838;margin:12px 0;"></div>',
                unsafe_allow_html=True)

    # Session stats
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Stitches", st.session_state.session_total)
    col_s2.metric("Runs", len(st.session_state.run_history))

    # Recording indicator
    is_rec = st.session_state.run_active
    ind_class = "status-live" if is_rec else "status-standby"
    ind_label = "Recording" if is_rec else "Standby"
    st.markdown(
        f'<div style="font-size:0.7em;color:#2a4050;margin-top:6px;">'
        f'<span class="status-indicator {ind_class}"></span>{ind_label}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="border-top:1px solid #1a2838;margin:12px 0;"></div>',
                unsafe_allow_html=True)

    if st.button("Reset Session", use_container_width=True):
        for k, v in [("run_active", False), ("run_stitches", []),
                     ("session_total", 0), ("log_lines", ["Session reset"]),
                     ("run_history", []), ("last_annotated", None),
                     ("last_centroids", []), ("live_running", False)]:
            st.session_state[k] = v
        st.session_state.centroid_history.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-top:20px;font-size:0.6em;color:#111e28;">
        Guneet Singh · Beta
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  PAGE: LIVE TRAINING
# ─────────────────────────────────────────────────────
if page == "Live Training":

    st.markdown("""
    <div class="page-title">Live Training</div>
    <div class="page-sub">Suture distance analysis · place dots in frame</div>
    """, unsafe_allow_html=True)

    col_cam, col_ctrl = st.columns([2.2, 1], gap="large")

    # ── RIGHT COLUMN: controls ──────────────────────────────────────────
    with col_ctrl:

        # ── MODE SELECTOR ──
        st.markdown('<div class="section-label">Camera Mode</div>', unsafe_allow_html=True)
        mode_col1, mode_col2 = st.columns(2)

        with mode_col1:
            snap_active = st.session_state.cam_mode == "snapshot"
            if st.button(
                "Snapshot" + (" ●" if snap_active else ""),
                key="btn_mode_snap",
                use_container_width=True,
                help="Take one photo and analyse it instantly"
            ):
                st.session_state.cam_mode    = "snapshot"
                st.session_state.live_running = False
                st.rerun()

        with mode_col2:
            live_active = st.session_state.cam_mode == "live"
            if st.button(
                "Live" + (" ●" if live_active else ""),
                key="btn_mode_live",
                use_container_width=True,
                help="Continuous feed — analyses every new capture automatically"
            ):
                st.session_state.cam_mode = "live"
                st.rerun()

        # Mode description card
        if st.session_state.cam_mode == "snapshot":
            st.markdown("""
            <div class="card" style="font-size:0.72em;color:#2a4a5a;margin-top:6px;">
              <b style="color:#4a7a8a;">Snapshot mode</b><br>
              Click the camera shutter button to capture one frame.
              The frame is analysed instantly and results appear below.
            </div>
            """, unsafe_allow_html=True)
        else:
            live_status = "running" if st.session_state.live_running else "paused"
            status_col = "#b85a5a" if st.session_state.live_running else "#2a4a5a"
            st.markdown(f"""
            <div class="card" style="font-size:0.72em;color:#2a4a5a;margin-top:6px;">
              <b style="color:{status_col};">Live mode · {live_status}</b><br>
              Camera captures continuously. Click the shutter to start,
              then use Start/Stop below to control the loop.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="border-top:1px solid #1a2838;margin:10px 0;"></div>',
                    unsafe_allow_html=True)

        # ── SESSION CONTROLS ──
        st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)

        run_col1, run_col2 = st.columns(2)
        with run_col1:
            if st.button("▶ Start Run",
                         disabled=st.session_state.run_active,
                         key="btn_start",
                         use_container_width=True):
                st.session_state.run_active   = True
                st.session_state.run_stitches = []
                st.session_state.centroid_history.clear()
                ts = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_lines.append(f"[{ts}] Run started")
                st.rerun()

        with run_col2:
            if st.button("⏹ End Run",
                         disabled=not st.session_state.run_active,
                         key="btn_end",
                         use_container_width=True):
                st.session_state.run_active   = False
                st.session_state.live_running = False
                run_s = st.session_state.run_stitches
                ts = datetime.now().strftime("%H:%M:%S")
                if run_s:
                    s_scores = [s["sc"] for s in run_s]
                    s_dists  = [s["d"]  for s in run_s]
                    optimal  = sum(1 for s in run_s if 60 <= s["d"] <= 120)
                    avg_run  = int(sum(s_scores)/max(1,len(s_scores)))
                    st.session_state.run_history.append({
                        "ts": ts, "stitches": len(run_s),
                        "avg_score": avg_run, "optimal": optimal,
                        "best": max(s_scores), "worst": min(s_scores),
                        "avg_dist": sum(s_dists)/max(1,len(s_dists)),
                    })
                    st.session_state.log_lines += [
                        "─" * 24,
                        f"[{ts}] Run complete — {len(run_s)} stitches, avg {avg_run}/100",
                        "─" * 24,
                    ]
                st.session_state.run_stitches = []
                st.rerun()

        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        # Log frame button — works in both modes
        if st.button("📌 Log This Frame",
                     disabled=not st.session_state.run_active,
                     key="btn_log",
                     use_container_width=True):
            n = log_frame()
            if n > 0:
                st.toast(f"{n} stitch(es) logged")
            st.rerun()

        st.markdown('<div style="border-top:1px solid #1a2838;margin:10px 0;"></div>',
                    unsafe_allow_html=True)

        # ── AUTO-HSV ──
        st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)

        hsv_labels = {"default": "Default", "auto": "Auto ✓", "manual": "Manual"}
        st.markdown(
            f'<div style="font-size:0.68em;color:#1e3a4a;margin-bottom:8px;">'
            f'Mode: <span style="color:#4a7a8a;">{hsv_labels.get(st.session_state.hsv_source,"?")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Auto-Tune HSV", key="btn_autotune", use_container_width=True):
            cam = st.session_state.get("main_camera")
            bgr = camera_to_bgr(cam)
            if bgr is None:
                st.warning("Capture a frame first, then click Auto-Tune.")
            else:
                with st.spinner("Analysing frame..."):
                    result = auto_tune_hsv(bgr)
                if result:
                    st.session_state.hsv = {k: result[k] for k in
                                            ("h_min","h_max","s_min","s_max","v_min","v_max")}
                    st.session_state.area = {"min_area": result["min_area"],
                                             "max_area": result["max_area"]}
                    st.session_state.hsv_source   = "auto"
                    st.session_state.auto_hsv_result = result
                    st.session_state.centroid_history.clear()
                    st.success(f"Done — hue peak {result['peak_hue']}°, {result['valid_blobs']} blobs")
                else:
                    st.error("Failed — check lighting and move closer.")

        if st.session_state.auto_hsv_result:
            r = st.session_state.auto_hsv_result
            with st.expander("Tune diagnostic"):
                st.markdown(f"""
                <div style="font-size:0.72em;color:#3a6070;line-height:1.9;">
                  Peak hue: {r['peak_hue']}° · spread: ±{r['spread']}<br>
                  H: {r['h_min']}–{r['h_max']} · S min: {r['s_min']} · V min: {r['v_min']}<br>
                  Area: {r['min_area']}–{r['max_area']}px² · blobs: {r['valid_blobs']}
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="border-top:1px solid #1a2838;margin:10px 0;"></div>',
                    unsafe_allow_html=True)

        # ── TIPS ──
        st.markdown('<div class="section-label">Setup</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7em;color:#1e3a4a;line-height:2;">
          Use round coloured stickers (blue, green, red)<br>
          Even lighting — avoid shadows on dots<br>
          Space dots 1–3 cm apart<br>
          Run Auto-Tune first<br>
          Optimal range = 60–120px
        </div>
        """, unsafe_allow_html=True)

    # ── LEFT COLUMN: camera + output ────────────────────────────────────
    with col_cam:

        # ────────────────────────────────────────────────────────────────
        #  SNAPSHOT MODE
        # ────────────────────────────────────────────────────────────────
        if st.session_state.cam_mode == "snapshot":

            # Single camera widget — user clicks shutter to capture
            snap_file = st.camera_input(
                label="Snapshot",
                label_visibility="collapsed",
                key="main_camera",
            )

            if snap_file is not None:
                # New photo taken — process it immediately
                pil_out, centroids = process_frame(snap_file)
                if pil_out:
                    st.image(pil_out, use_container_width=True, caption=None)
                render_score_card(st.session_state.last_centroids)

            elif st.session_state.last_annotated is not None:
                # No new photo yet — show last processed result
                st.image(bgr_to_pil(st.session_state.last_annotated),
                         use_container_width=True, caption=None)
                render_score_card(st.session_state.last_centroids)
            else:
                # First load — show placeholder
                st.markdown("""
                <div class="card" style="text-align:center;padding:40px 20px;color:#1e3a4a;">
                  Click the shutter button above to take a photo<br>
                  <span style="font-size:0.8em;color:#162838;">
                    Dots will be detected and analysed instantly
                  </span>
                </div>
                """, unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────
        #  LIVE MODE
        #  Streamlit doesn't support true video streaming, so we simulate
        #  it by auto-calling st.rerun() every ~0.5s while live_running=True.
        #  The camera_input widget re-captures each rerun cycle.
        # ────────────────────────────────────────────────────────────────
        else:
            # Start / Stop live loop buttons
            live_btn_col1, live_btn_col2 = st.columns(2)
            with live_btn_col1:
                if st.button(
                    "▶ Start Live",
                    disabled=st.session_state.live_running,
                    key="btn_live_start",
                    use_container_width=True,
                ):
                    st.session_state.live_running = True
                    st.rerun()

            with live_btn_col2:
                if st.button(
                    "⏹ Stop Live",
                    disabled=not st.session_state.live_running,
                    key="btn_live_stop",
                    use_container_width=True,
                ):
                    st.session_state.live_running = False
                    st.rerun()

            # Live status bar
            if st.session_state.live_running:
                st.markdown("""
                <div style="font-size:0.68em;color:#6a3030;letter-spacing:0.08em;margin:4px 0 8px 0;">
                  <span class="status-indicator status-live"></span>LIVE · capturing continuously
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-size:0.68em;color:#1e3040;letter-spacing:0.08em;margin:4px 0 8px 0;">
                  <span class="status-indicator status-standby"></span>Stopped
                </div>
                """, unsafe_allow_html=True)

            # Camera widget — always present in live mode so browser keeps stream open
            live_file = st.camera_input(
                label="Live feed",
                label_visibility="collapsed",
                key="main_camera",
            )

            if live_file is not None:
                # Process every captured frame
                pil_out, centroids = process_frame(live_file)
                if pil_out:
                    st.image(pil_out, use_container_width=True, caption=None)
                render_score_card(st.session_state.last_centroids)

            elif st.session_state.last_annotated is not None:
                # Show last frame while waiting for next capture
                st.image(bgr_to_pil(st.session_state.last_annotated),
                         use_container_width=True, caption=None)
                render_score_card(st.session_state.last_centroids)

            # Auto-rerun loop: if live is running, wait 0.45s then rerun
            # This forces Streamlit to re-execute the script, which re-triggers
            # the camera widget and processes the next frame.
            if st.session_state.live_running:
                time.sleep(0.45)
                st.rerun()

# ─────────────────────────────────────────────────────
#  PAGE: SESSION HISTORY
# ─────────────────────────────────────────────────────
elif page == "Session History":

    st.markdown("""
    <div class="page-title">Session History</div>
    <div class="page-sub">Run summaries · score trends · event log</div>
    """, unsafe_allow_html=True)

    if not st.session_state.run_history and not st.session_state.log_lines:
        st.markdown("""
        <div class="card" style="text-align:center;padding:40px;color:#1e3a4a;">
          No data yet — complete a run on the Live Training page.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Stitches", st.session_state.session_total)
        m2.metric("Runs", len(st.session_state.run_history))
        if st.session_state.run_history:
            avg_scores = [r["avg_score"] for r in st.session_state.run_history]
            m3.metric("Avg Score", f"{int(sum(avg_scores)/len(avg_scores))}/100")
            opt_rates  = [r["optimal"]/max(1,r["stitches"])*100 for r in st.session_state.run_history]
            m4.metric("Optimal Rate", f"{int(sum(opt_rates)/len(opt_rates))}%")
        else:
            m3.metric("Avg Score", "—")
            m4.metric("Optimal Rate", "—")

        st.markdown('<div style="border-top:1px solid #1a2838;margin:16px 0;"></div>',
                    unsafe_allow_html=True)

        # Score trend chart
        if st.session_state.run_history:
            st.markdown('<div class="section-label">Score by Run</div>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame({
                "Run": [f"Run {i+1}" for i in range(len(st.session_state.run_history))],
                "Avg Score": [r["avg_score"] for r in st.session_state.run_history],
            }).set_index("Run")
            st.bar_chart(df, color="#4a7a8a", height=200)

            st.markdown('<div style="border-top:1px solid #1a2838;margin:16px 0;"></div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="section-label">Run Summaries</div>', unsafe_allow_html=True)
            for i, r in enumerate(reversed(st.session_state.run_history)):
                n = len(st.session_state.run_history) - i
                opt_pct = int(r["optimal"]/max(1,r["stitches"])*100)
                with st.expander(f"Run {n}  ·  {r['ts']}  ·  {r['avg_score']}/100"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Stitches", r["stitches"])
                    c2.metric("Optimal", f"{r['optimal']}/{r['stitches']} ({opt_pct}%)")
                    c3.metric("Avg Dist", f"{r['avg_dist']:.1f}px")
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Best", f"{r['best']}/100")
                    cc2.metric("Worst", f"{r['worst']}/100")

        # Event log
        if st.session_state.log_lines:
            st.markdown('<div style="border-top:1px solid #1a2838;margin:16px 0;"></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-label">Event Log</div>', unsafe_allow_html=True)
            lines_html = "".join(
                f'<div class="log-line">{l}</div>'
                for l in reversed(st.session_state.log_lines[-60:])
            )
            st.markdown(f'<div class="log-box">{lines_html}</div>', unsafe_allow_html=True)

        # Export
        st.markdown('<div style="border-top:1px solid #1a2838;margin:16px 0;"></div>',
                    unsafe_allow_html=True)
        if st.button("Export Session JSON"):
            payload = {
                "exported": datetime.now().isoformat(),
                "session_total": st.session_state.session_total,
                "run_history":   st.session_state.run_history,
                "log":           st.session_state.log_lines,
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(payload, indent=2),
                file_name=f"suterup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

# ─────────────────────────────────────────────────────
#  PAGE: CALIBRATION
# ─────────────────────────────────────────────────────
elif page == "Calibration":

    st.markdown("""
    <div class="page-title">Calibration</div>
    <div class="page-sub">Manual HSV control · mask preview</div>
    """, unsafe_allow_html=True)

    col_sl, col_prev = st.columns([1, 1.6], gap="large")

    with col_sl:
        st.markdown('<div class="section-label">HSV Range</div>', unsafe_allow_html=True)
        h = st.session_state.hsv

        # Each slider: H is 0-179 (OpenCV half-degree scale), S/V are 0-255
        h_min = st.slider("Hue min",         0, 179, h["h_min"], 1)
        h_max = st.slider("Hue max",         0, 179, h["h_max"], 1)
        s_min = st.slider("Saturation min",  0, 255, h["s_min"], 1)
        s_max = st.slider("Saturation max",  0, 255, h["s_max"], 1)
        v_min = st.slider("Brightness min",  0, 255, h["v_min"], 1)
        v_max = st.slider("Brightness max",  0, 255, h["v_max"], 1)

        st.markdown('<div class="section-label" style="margin-top:12px;">Blob Area (px²)</div>',
                    unsafe_allow_html=True)
        a = st.session_state.area
        min_a = st.slider("Min area",  10, 3000,  a["min_area"], 10)
        max_a = st.slider("Max area", 200, 25000, a["max_area"], 100)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Apply", use_container_width=True):
                st.session_state.hsv = dict(
                    h_min=h_min, h_max=h_max,
                    s_min=s_min, s_max=s_max,
                    v_min=v_min, v_max=v_max,
                )
                st.session_state.area = dict(min_area=min_a, max_area=max_a)
                st.session_state.hsv_source = "manual"
                st.session_state.centroid_history.clear()
                st.success("Applied")
                st.rerun()
        with col_b2:
            if st.button("Defaults", use_container_width=True):
                st.session_state.hsv  = dict(h_min=90, h_max=130, s_min=80, s_max=255, v_min=50, v_max=255)
                st.session_state.area = dict(min_area=200, max_area=8000)
                st.session_state.hsv_source = "default"
                st.session_state.centroid_history.clear()
                st.rerun()

        st.markdown("""
        <div style="margin-top:14px;font-size:0.68em;color:#1a3040;line-height:2;">
          Red: 0–10 or 160–179<br>
          Orange: 10–25 · Yellow: 25–35<br>
          Green: 35–85 · Cyan: 85–100<br>
          Blue: 100–130 · Purple: 130–160
        </div>
        """, unsafe_allow_html=True)

    with col_prev:
        st.markdown('<div class="section-label">Mask Preview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7em;color:#1e3a4a;margin-bottom:10px;">
          Capture a frame — white areas show what the algorithm detects.
          Adjust sliders until only the dots appear white.
        </div>
        """, unsafe_allow_html=True)

        cal_cam = st.camera_input("Cal camera", label_visibility="collapsed", key="cal_camera")
        if cal_cam is not None:
            bgr = camera_to_bgr(cal_cam)
            if bgr is not None:
                try:
                    # Build mask with current slider values (not yet applied to state)
                    mask = _build_mask_internal(bgr, h_min, h_max, s_min, s_max, v_min, v_max)
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    # Stack original + mask side by side
                    combined = np.hstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask_rgb])
                    st.image(combined, use_container_width=True,
                             caption="Original (left) · Detected mask (right)")

                    # Count valid blobs with current slider settings
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = sum(
                        1 for c in cnts
                        if min_a <= cv2.contourArea(c) <= max_a
                        and cv2.arcLength(c, True) > 0
                        and (4*math.pi*cv2.contourArea(c)/cv2.arcLength(c,True)**2) >= 0.20
                    )
                    st.markdown(
                        f'<div style="font-size:0.76em;color:#4a7a8a;margin-top:6px;">'
                        f'{valid} valid blob(s) with current settings</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Mask error: {e}")

# ─────────────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────────────
elif page == "About":

    st.markdown("""
    <div class="page-title">About</div>
    <div class="page-sub">Scoring guide · tech stack · version notes</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Scoring Guide", "Tech Stack", "Version Notes"])

    with tab1:
        st.markdown("""
        <div class="card">
          <div style="font-size:0.82em;font-weight:600;color:#c8dce8;
                      margin-bottom:12px;letter-spacing:0.06em;">
            Suture Scoring Rubric
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:0.76em;">
            <tr style="color:#1a3040;border-bottom:1px solid #0e1820;">
              <td style="padding:6px 10px;">Range</td>
              <td style="padding:6px 10px;">Grade</td>
              <td style="padding:6px 10px;">Score</td>
              <td style="padding:6px 10px;">Clinical note</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;color:#3a6070;">&lt; 40px</td>
              <td class="grade-danger" style="padding:6px 10px;">Too Close</td>
              <td style="padding:6px 10px;color:#3a6070;">0–55</td>
              <td style="padding:6px 10px;color:#2a4050;">Risk of tissue necrosis</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;color:#3a6070;">40–60px</td>
              <td class="grade-warn" style="padding:6px 10px;">Short</td>
              <td style="padding:6px 10px;color:#3a6070;">55–77</td>
              <td style="padding:6px 10px;color:#2a4050;">Slightly under-spaced</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;color:#3a6070;">60–120px</td>
              <td class="grade-optimal" style="padding:6px 10px;">Optimal ✓</td>
              <td style="padding:6px 10px;color:#3a6070;">77–100</td>
              <td style="padding:6px 10px;color:#2a4050;">Ideal spacing</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;color:#3a6070;">120–160px</td>
              <td class="grade-warn" style="padding:6px 10px;">Wide</td>
              <td style="padding:6px 10px;color:#3a6070;">55–77</td>
              <td style="padding:6px 10px;color:#2a4050;">Wound gap risk</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;color:#3a6070;">&gt; 160px</td>
              <td class="grade-danger" style="padding:6px 10px;">Too Wide</td>
              <td style="padding:6px 10px;color:#3a6070;">0–55</td>
              <td style="padding:6px 10px;color:#2a4050;">Closure failure risk</td>
            </tr>
          </table>
          <div style="margin-top:12px;font-size:0.7em;color:#1a3040;">
            Pixel distances calibrated for webcam at ~25–35cm. Score peak at 90px.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="card" style="font-size:0.76em;line-height:2.2;color:#3a6070;">
          <b style="color:#7aaac8;">Streamlit</b> — web framework, state management, camera widget<br>
          <b style="color:#7aaac8;">OpenCV</b> — HSV masking, contour detection, morphology, annotation<br>
          <b style="color:#7aaac8;">NumPy</b> — pixel array math, histogram analysis<br>
          <b style="color:#7aaac8;">Pillow</b> — BGR→RGB conversion for Streamlit display<br>
          <b style="color:#7aaac8;">Pandas</b> — score trend chart on History page<br>
          <b style="color:#7aaac8;">Python stdlib</b> — math, json, collections, datetime, io
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="card" style="font-size:0.76em;line-height:2.2;color:#3a6070;">
          <b style="color:#c8dce8;">V5.1</b> — Two-mode camera (Snapshot/Live), cleaner UI, no flash/gradients<br>
          <b style="color:#3a6070;">V5.0</b> — Streamlit migration, multi-page nav, mask preview<br>
          <b style="color:#3a6070;">V4.0</b> — Auto-HSV 6-pass, temporal smoothing, Gradio live stream<br>
          <b style="color:#3a6070;">V3.x</b> — Manual HSV, basic blob detection, static image upload
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;font-size:0.6em;color:#0e1a24;padding:20px 0 0 0;">
      Suterup V5.1 · Guneet Singh · Beta
    </div>
    """, unsafe_allow_html=True)
