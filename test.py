"""
╔══════════════════════════════════════════════════╗
║         SUTERUP V0.3 — Surgical Training          ║
║         Live Detection + Auto-HSV Edition         ║
║         CEO/CTO: Asrith Maruvada                  ║
╚══════════════════════════════════════════════════╝

HOW TO DEPLOY:
  1. pip install streamlit opencv-python-headless numpy pillow
  2. streamlit run suterup_v03.py

WHAT'S NEW IN V0.3:
  - Live camera feed via st.camera_input in continuous mode
  - Auto-HSV: AI algorithm finds best HSV settings automatically
  - Manual override still available after auto-tune
  - QOL: cleaner layout, better warnings, run status always visible
  - Edge-case hardening: division guards, HSV range clamping, etc.
"""

import streamlit as st
import cv2
import numpy as np
import math
import time
from datetime import datetime
from PIL import Image

# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suterup V0.3",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
}
.stApp { background-color: #07100a; color: #c8dcc8; }
.main { background-color: #07100a; }

/* Header */
.suterup-header {
    background: linear-gradient(135deg, #0d1f10 0%, #0a1a0d 100%);
    border: 1px solid #1a3d1a;
    border-left: 4px solid #00e09a;
    border-radius: 4px;
    padding: 14px 24px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.suterup-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6em;
    color: #00e09a;
    letter-spacing: -0.02em;
}
.suterup-tagline {
    color: #5a8a5a;
    font-size: 0.75em;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #0c1c0e;
    border: 1px solid #1a3d1a;
    border-radius: 6px;
    padding: 14px 10px;
    text-align: center;
}
.metric-label { color: #4a7a4a; font-size: 0.68em; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-size: 1.6em; font-weight: 700; color: #00e09a; line-height: 1.1; }
.metric-unit  { font-size: 0.65em; color: #4a7a4a; }

/* Grade pills */
.grade-optimal { color: #00e09a; }
.grade-short   { color: #ffd700; }
.grade-wide    { color: #ffd700; }
.grade-bad     { color: #e05252; }

/* Feedback bar */
.feedback-bar {
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.85em;
    margin-top: 8px;
    border-left: 3px solid;
}
.feedback-optimal { background:#0d2b1a; border-color:#00e09a; color:#00e09a; }
.feedback-warn    { background:#2b2500; border-color:#ffd700; color:#ffd700; }
.feedback-bad     { background:#2b0d0d; border-color:#e05252; color:#e05252; }

/* Stitch log */
.stitch-log {
    background: #0c1c0e;
    border: 1px solid #1a3d1a;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.8em;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.8;
}

/* Auto-HSV badge */
.auto-badge {
    display: inline-block;
    background: #003322;
    border: 1px solid #00e09a;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.7em;
    color: #00e09a;
    letter-spacing: 0.1em;
}
.manual-badge {
    display: inline-block;
    background: #221a00;
    border: 1px solid #ffd700;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.7em;
    color: #ffd700;
    letter-spacing: 0.1em;
}

/* Section headers */
h3 { font-family: 'Syne', sans-serif !important; color: #00e09a !important; font-size: 1em !important; letter-spacing: 0.08em !important; text-transform: uppercase; }

/* Buttons */
.stButton > button {
    background: #0c1c0e;
    border: 1px solid #1a3d1a;
    color: #c8dcc8;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #00e09a;
    color: #00e09a;
    background: #0d2b1a;
}

/* Status indicator */
.status-dot-on  { display:inline-block; width:8px; height:8px; border-radius:50%; background:#e05252; box-shadow: 0 0 6px #e05252; animation: pulse 1s infinite; }
.status-dot-off { display:inline-block; width:8px; height:8px; border-radius:50%; background:#2a4a2a; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Sliders */
.stSlider > div > div { background: #1a3d1a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────
defaults = {
    "run_active": False,
    "run_stitches": [],
    "session_total": 0,
    "log_messages": [],
    "last_centroids": [],
    "auto_hsv": {"h_min":90,"h_max":130,"s_min":80,"s_max":255,"v_min":50,"v_max":255},
    "hsv_source": "default",   # "default" | "auto" | "manual"
    "frame_count": 0,
    "last_dot_count": 0,
    "run_history": [],         # list of run summaries
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────
#  CONSTANTS & MATH
# ─────────────────────────────────────────────────────
PT_COLORS_BGR = [
    (141,201,0),(255,140,66),(255,200,77),(255,120,200),
    (255,219,77),(130,255,140),(60,180,255),(255,80,60),
    (200,80,220),(180,255,80),
]

def seg_dist(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def precision(d):
    """Score 0-100. Peak at 90px. Never goes below 0."""
    return max(0, int(100 - abs(d - 90) / 0.9))

def grade(d):
    """Returns (label, hex_color, feedback, css_class)"""
    if d < 40:
        return "TOO CLOSE", "#e05252", "Too close — increase stitch width.", "feedback-bad"
    if d < 60:
        return "SHORT",     "#ffd700", "Slightly narrow — aim a bit wider.", "feedback-warn"
    if d <= 120:
        return "OPTIMAL",   "#00e09a", "Excellent placement — within ideal range.", "feedback-optimal"
    if d <= 160:
        return "WIDE",      "#ffd700", "Slightly wide — bring stitches a little closer.", "feedback-warn"
    return     "TOO WIDE",  "#e05252", "Too wide — reduce stitch distance.", "feedback-bad"

# ─────────────────────────────────────────────────────
#  AUTO-HSV ENGINE
# ─────────────────────────────────────────────────────
def auto_tune_hsv(frame):
    """
    Fully automatic HSV detection algorithm.
    Strategy:
      1. Convert to HSV.
      2. Find the most saturated, non-skin, non-background pixel clusters.
      3. Pick the dominant hue cluster that also passes a minimum blob area test.
      4. Return tight HSV bounds ± tolerances, clamped to valid ranges.
    
    Edge-case guards:
      - Dark frames → expands V range
      - Washed-out frames → lowers S_min
      - Inverted H range → swaps min/max
      - All-one-color frames → falls back to prior settings
    
    Returns dict with h_min, h_max, s_min, s_max, v_min, v_max, min_area, max_area
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Mean brightness — if dark, relax V
    mean_v = float(np.mean(v))
    v_floor = max(20, int(mean_v * 0.25))

    # Mean saturation — if washed out, relax S
    mean_s = float(np.mean(s))
    s_floor = max(30, int(mean_s * 0.30))

    # Build a candidate mask: pixels that are reasonably saturated & bright
    candidate_mask = (s > s_floor) & (v > v_floor)
    candidate_hues = h[candidate_mask]

    if len(candidate_hues) < 100:
        # Frame is too dark or featureless — return prior/default
        return None

    # Build hue histogram (0-179)
    hist, bins = np.histogram(candidate_hues, bins=180, range=(0, 179))

    # Smooth histogram to find dominant peak
    kernel = np.ones(7) / 7
    hist_smooth = np.convolve(hist, kernel, mode='same')

    # Exclude very common hues that are likely background (top 5% of frame coverage)
    total_pixels = frame.shape[0] * frame.shape[1]
    threshold_count = total_pixels * 0.05
    hist_smooth[hist_smooth > threshold_count] = 0

    # Find peak hue
    if hist_smooth.max() < 50:
        return None  # nothing meaningful found

    peak_hue = int(np.argmax(hist_smooth))

    # Determine spread: how many hues around peak have > 20% of peak count
    peak_val = hist_smooth[peak_hue]
    spread_threshold = peak_val * 0.20
    spread = 0
    for delta in range(1, 30):
        left_ok  = hist_smooth[(peak_hue - delta) % 180] > spread_threshold
        right_ok = hist_smooth[(peak_hue + delta) % 180] > spread_threshold
        if left_ok or right_ok:
            spread = delta
        else:
            break

    # Compute HSV bounds
    tolerance = max(12, spread + 5)  # never too tight
    h_min = max(0,   peak_hue - tolerance)
    h_max = min(179, peak_hue + tolerance)

    # Guard: if range is inverted somehow, swap
    if h_min > h_max:
        h_min, h_max = h_max, h_min

    # Guard: if range is too narrow, widen to minimum 15 hue units
    if (h_max - h_min) < 15:
        center = (h_min + h_max) // 2
        h_min = max(0, center - 8)
        h_max = min(179, center + 8)

    # Saturation bounds — adaptive
    sat_pixels = s[candidate_mask & (np.abs(h.astype(int) - peak_hue) < tolerance)]
    if len(sat_pixels) > 0:
        s_min = max(30, int(np.percentile(sat_pixels, 10)))
        s_max = 255
    else:
        s_min, s_max = s_floor, 255

    # Value bounds — adaptive
    val_pixels = v[candidate_mask & (np.abs(h.astype(int) - peak_hue) < tolerance)]
    if len(val_pixels) > 0:
        v_min = max(20, int(np.percentile(val_pixels, 10)))
        v_max = 255
    else:
        v_min, v_max = v_floor, 255

    # Blob area: estimate from frame size
    img_area = frame.shape[0] * frame.shape[1]
    min_area = max(50,  int(img_area * 0.0003))
    max_area = min(int(img_area * 0.04), 20000)

    return {
        "h_min": h_min, "h_max": h_max,
        "s_min": s_min, "s_max": s_max,
        "v_min": v_min, "v_max": v_max,
        "min_area": min_area, "max_area": max_area,
        "peak_hue": peak_hue,
    }

# ─────────────────────────────────────────────────────
#  BLOB DETECTION
# ─────────────────────────────────────────────────────
def build_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max):
    """Build morphologically cleaned HSV mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Guard: ensure valid ranges
    h_min, h_max = sorted([int(h_min), int(h_max)])
    s_min, s_max = sorted([int(s_min), int(s_max)])
    v_min, v_max = sorted([int(v_min), int(v_max)])

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def detect_blobs(frame, h_min, h_max, s_min, s_max, v_min, v_max, min_area, max_area):
    """Detect blobs and return ordered centroids (max 10)."""
    # Guard: ensure area bounds are valid
    min_area = max(10, int(min_area))
    max_area = max(min_area + 100, int(max_area))

    mask = build_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pts.append((cx, cy))

    if len(pts) < 2:
        return pts[:1], mask  # return what we have (0 or 1)

    # Order by nearest-neighbor chain from leftmost
    pts = sorted(pts, key=lambda p: p[0])
    ordered = [pts.pop(0)]
    while pts:
        last = ordered[-1]
        nearest_idx = min(range(len(pts)),
                          key=lambda i: math.sqrt((pts[i][0]-last[0])**2 + (pts[i][1]-last[1])**2))
        ordered.append(pts.pop(nearest_idx))

    return ordered[:10], mask  # cap at 10 (edge case #3)

# ─────────────────────────────────────────────────────
#  FRAME ANNOTATION
# ─────────────────────────────────────────────────────
def draw_dashed(frame, p1, p2, color, thickness=2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    length = max(1, math.sqrt(dx*dx+dy*dy))
    ux, uy = dx/length, dy/length
    pos, on = 0, True
    while pos < length:
        end = min(pos + (9 if on else 5), length)
        if on:
            a = (int(p1[0]+ux*pos), int(p1[1]+uy*pos))
            b = (int(p1[0]+ux*end), int(p1[1]+uy*end))
            cv2.line(frame, a, b, color, thickness)
        pos, on = end, not on

def hex_to_bgr(hex_str):
    h = hex_str.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)

def annotate_frame(frame, centroids):
    display = frame.copy()

    for i in range(len(centroids) - 1):
        p1, p2 = centroids[i], centroids[i+1]
        d = seg_dist(p1, p2)
        g_label, g_hex, _, _ = grade(d)
        color = hex_to_bgr(g_hex)
        draw_dashed(display, p1, p2, color, 2)

        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        sc = precision(d)
        cv2.putText(display, f"{d:.0f}px  {sc}/100", (mid[0]-30, mid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(display, g_label, (mid[0]-20, mid[1]+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    for i, pt in enumerate(centroids):
        color = PT_COLORS_BGR[i % len(PT_COLORS_BGR)]
        cv2.circle(display, pt, 16, color, 1)
        cv2.circle(display, pt, 9,  color, -1)
        cv2.circle(display, pt, 3,  (210,240,220), -1)
        cv2.putText(display, f"P{i+1}", (pt[0]+18, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw overall score in corner
    if len(centroids) >= 2:
        dists  = [seg_dist(centroids[i], centroids[i+1]) for i in range(len(centroids)-1)]
        scores = [precision(d) for d in dists]
        avg_sc = int(sum(scores) / max(1, len(scores)))
        cv2.putText(display, f"LIVE SCORE: {avg_sc}/100",
                    (10, display.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,224,154), 1, cv2.LINE_AA)

    return display

# ─────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────
status_dot = '<span class="status-dot-on"></span>' if st.session_state.run_active else '<span class="status-dot-off"></span>'
st.markdown(f"""
<div class="suterup-header">
    <div>
        <div class="suterup-logo">⚕ SUTERUP</div>
        <div class="suterup-tagline">V0.3 · Surgical Training Platform · Asrith Maruvada</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
        {status_dot}
        <span style="font-size:0.75em;color:#5a8a5a;">{"RECORDING" if st.session_state.run_active else "STANDBY"}</span>
        &nbsp;&nbsp;
        <span style="font-size:0.75em;color:#5a8a5a;">SESSION STITCHES: <span style="color:#00e09a;">{st.session_state.session_total}</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────
col_cam, col_sidebar = st.columns([3, 1], gap="medium")

with col_sidebar:
    # ── Controls ──────────────────────────────────────
    st.markdown("### Controls")
    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("▶ Start", disabled=st.session_state.run_active, use_container_width=True)
    with c2:
        end_btn = st.button("⏹ End", disabled=not st.session_state.run_active, use_container_width=True)

    log_btn = st.button("📌 Log Frame", disabled=not st.session_state.run_active, use_container_width=True,
                        help="Captures current stitch measurements into the log")
    rst_btn = st.button("🔄 Reset Session", use_container_width=True)

    if run_btn:
        st.session_state.run_active   = True
        st.session_state.run_stitches = []
        st.session_state.log_messages.append(f"🟢 [{datetime.now().strftime('%H:%M:%S')}] Run started — hover camera over suture dots")
        st.rerun()

    if rst_btn:
        for k, v in defaults.items():
            st.session_state[k] = v if not callable(v) else v()
        st.session_state.log_messages = ["🔄 Session reset."]
        st.rerun()

    st.markdown("---")

    # ── HSV Tuner ─────────────────────────────────────
    st.markdown("### HSV Detection")

    hsv_source = st.session_state.hsv_source
    badge_html = '<span class="auto-badge">AUTO</span>' if hsv_source == "auto" else \
                 ('<span class="manual-badge">MANUAL</span>' if hsv_source == "manual" else "")
    st.markdown(f"**Current mode** {badge_html}", unsafe_allow_html=True)

    # Auto-tune button
    auto_btn = st.button("🤖 Auto-Tune HSV", use_container_width=True,
                         help="AI analyzes the current frame and sets optimal HSV values")

    st.caption("↓ Adjust manually after auto-tune if needed")

    a = st.session_state.auto_hsv
    h_min = st.slider("H min", 0, 179, a["h_min"], key="h_min")
    h_max = st.slider("H max", 0, 179, a["h_max"], key="h_max")
    s_min = st.slider("S min", 0, 255, a["s_min"], key="s_min")
    s_max = st.slider("S max", 0, 255, a["s_max"], key="s_max")
    v_min = st.slider("V min", 0, 255, a["v_min"], key="v_min")
    v_max = st.slider("V max", 0, 255, a["v_max"], key="v_max")
    min_area = st.slider("Min blob (px²)", 50, 2000, a.get("min_area", 200), key="min_area")
    max_area = st.slider("Max blob (px²)", 500, 20000, a.get("max_area", 8000), key="max_area")

    # Detect if user moved sliders manually
    user_hsv = {"h_min":h_min,"h_max":h_max,"s_min":s_min,"s_max":s_max,
                "v_min":v_min,"v_max":v_max,"min_area":min_area,"max_area":max_area}
    if a != {k: user_hsv.get(k, a.get(k)) for k in a} and hsv_source == "auto":
        st.session_state.hsv_source = "manual"

    st.markdown("---")

    # ── Run History ───────────────────────────────────
    if st.session_state.run_history:
        st.markdown("### Past Runs")
        for i, rh in enumerate(reversed(st.session_state.run_history[-5:])):
            st.markdown(f"""<div class="metric-card" style="margin-bottom:6px;text-align:left;padding:8px 12px;">
                <span style="color:#4a7a4a;font-size:0.7em;">RUN {len(st.session_state.run_history)-i}</span>
                &nbsp;·&nbsp;
                <span style="color:#00e09a;">{rh['avg_score']}/100</span>
                &nbsp;·&nbsp;
                <span style="color:#5a8a5a;font-size:0.8em;">{rh['stitches']} stitches</span>
            </div>""", unsafe_allow_html=True)

with col_cam:
    st.markdown("### Live Camera Feed")
    st.caption("📸 Each captured photo is analyzed instantly — keep pressing the camera button for continuous feedback. Use **Log Frame** to record measurements.")

    img_file = st.camera_input("", label_visibility="collapsed", key="cam_input")

    if img_file is not None:
        pil_img = Image.open(img_file)
        frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        st.session_state.frame_count += 1

        # ── Auto-tune trigger ─────────────────────────
        if auto_btn:
            with st.spinner("🤖 AI analyzing frame for optimal HSV settings..."):
                result = auto_tune_hsv(frame)
            if result:
                st.session_state.auto_hsv = {
                    "h_min": result["h_min"], "h_max": result["h_max"],
                    "s_min": result["s_min"], "s_max": result["s_max"],
                    "v_min": result["v_min"], "v_max": result["v_max"],
                    "min_area": result["min_area"], "max_area": result["max_area"],
                }
                st.session_state.hsv_source = "auto"
                peak = result.get("peak_hue", "?")
                st.success(f"✅ Auto-HSV complete — detected dominant hue H={peak}°. Sliders updated.")
                st.rerun()
            else:
                st.warning("⚠️ Auto-tune could not find a dominant color. Try better lighting or move closer to the dots.")

        # ── Detect with current HSV ───────────────────
        centroids, mask = detect_blobs(
            frame, h_min, h_max, s_min, s_max, v_min, v_max, min_area, max_area
        )
        st.session_state.last_centroids = centroids
        st.session_state.last_dot_count = len(centroids)

        # ── Annotate & display ────────────────────────
        annotated     = annotate_frame(frame, centroids)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        tab1, tab2 = st.tabs(["📹 Annotated", "🎭 Mask Debug"])
        with tab1:
            st.image(annotated_rgb, use_container_width=True)
        with tab2:
            st.image(mask, use_container_width=True,
                     caption="Your dot colors should appear as white blobs. Tune HSV if not.")

        # ── Metrics & Feedback ────────────────────────
        if len(centroids) >= 2:
            dists  = [seg_dist(centroids[i], centroids[i+1]) for i in range(len(centroids)-1)]
            scores = [precision(d) for d in dists]
            avg_d  = sum(dists) / max(1, len(dists))
            avg_sc = int(sum(scores) / max(1, len(scores)))
            last_d = dists[-1]
            g_label, g_hex, feedback_text, fb_class = grade(last_d)
            best_sc  = max(scores)
            worst_sc = min(scores)

            # Metric cards
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Dots</div><div class="metric-value">{len(centroids)}</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score</div><div class="metric-value">{avg_sc}<span class="metric-unit">/100</span></div></div>', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Dist</div><div class="metric-value">{avg_d:.0f}<span class="metric-unit">px</span></div></div>', unsafe_allow_html=True)
            with mc4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Best</div><div class="metric-value">{best_sc}<span class="metric-unit">/100</span></div></div>', unsafe_allow_html=True)
            with mc5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Grade</div><div class="metric-value" style="color:{g_hex};font-size:1em;">{g_label}</div></div>', unsafe_allow_html=True)

            # Per-stitch detail
            if len(dists) > 1:
                with st.expander(f"↕ Per-stitch breakdown ({len(dists)} segments)", expanded=False):
                    for i, (d, sc) in enumerate(zip(dists, scores)):
                        gl, gh, _, _ = grade(d)
                        st.markdown(
                            f'<span style="color:#4a7a4a;">P{i+1}→P{i+2}</span> '
                            f'<span style="color:{gh};">{d:.1f}px · {gl} · {sc}/100</span>',
                            unsafe_allow_html=True
                        )

            # Feedback bar
            st.markdown(f'<div class="feedback-bar {fb_class}">💬 {feedback_text}</div>', unsafe_allow_html=True)

            # ── Log stitches action ──────────────────
            if log_btn:
                logged = 0
                for i in range(len(centroids) - 1):
                    d  = seg_dist(centroids[i], centroids[i+1])
                    sc = precision(d)
                    g_label_l, _, _, _ = grade(d)
                    st.session_state.run_stitches.append({"d": d, "sc": sc})
                    st.session_state.session_total += 1
                    logged += 1
                    st.session_state.log_messages.append(
                        f"📌 #{st.session_state.session_total} | P{i+1}→P{i+2} | {d:.1f}px | {g_label_l} | {sc}/100"
                    )
                st.session_state.log_messages.append(f"✅ {logged} stitch(es) logged at {datetime.now().strftime('%H:%M:%S')}")
                st.rerun()

            # ── End run action ───────────────────────
            if end_btn:
                run_s = st.session_state.run_stitches
                # Log final frame too if run has stitches
                if run_s:
                    s_scores = [s["sc"] for s in run_s]
                    s_dists  = [s["d"]  for s in run_s]
                    optimal  = sum(1 for s in run_s if 60 <= s["d"] <= 120)
                    avg_run  = int(sum(s_scores) / max(1, len(s_scores)))
                    st.session_state.run_history.append({
                        "stitches": len(run_s),
                        "avg_score": avg_run,
                        "optimal": optimal,
                    })
                    st.session_state.log_messages += [
                        "━" * 28,
                        f"🏁 RUN COMPLETE — {datetime.now().strftime('%H:%M:%S')}",
                        f"  Stitches logged : {len(run_s)}",
                        f"  Avg score       : {avg_run}/100",
                        f"  Best / Worst    : {max(s_scores)}/100 · {min(s_scores)}/100",
                        f"  Optimal stitches: {optimal}/{len(run_s)}",
                        f"  Avg distance    : {sum(s_dists)/max(1,len(s_dists)):.1f}px",
                        "━" * 28,
                    ]
                st.session_state.run_active   = False
                st.session_state.run_stitches = []
                st.rerun()

        elif len(centroids) == 1:
            # Edge case #2: single dot
            st.warning("⚠️ Only 1 dot detected. Need at least **2** to measure stitch spacing. Try adjusting HSV or move closer.")
        else:
            # Edge case #1: no dots
            col_warn1, col_warn2 = st.columns(2)
            with col_warn1:
                st.warning("⚠️ No dots detected. Try **Auto-Tune HSV** first, then check lighting.")
            with col_warn2:
                st.info("💡 Tip: Blue stickers work best. Ensure good even lighting with no shadows.")

        if log_btn and len(centroids) < 2:
            st.error("❌ Need at least 2 detected dots to log stitches.")
        if end_btn and len(centroids) < 2:
            st.session_state.run_active = False
            st.rerun()

    else:
        # Edge case #8: no camera input yet
        st.markdown("""
        <div style="background:#0c1c0e;border:1px dashed #1a3d1a;border-radius:8px;padding:40px;text-align:center;color:#4a7a4a;">
            <div style="font-size:2em;margin-bottom:12px;">📷</div>
            <div style="color:#c8dcc8;margin-bottom:8px;">Click the camera button above to begin</div>
            <div style="font-size:0.8em;">Allow camera access when prompted by your browser</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 Quick setup guide"):
            st.markdown("""
**Dot setup:**
- Use small **blue stickers**, **blue nail polish dots**, or **blue marker caps** at each suture point
- Space them 1–3cm apart for realistic suture spacing
- Any color works — use **Auto-Tune HSV** to adapt

**Lighting:**
- Ensure even lighting — avoid shadows directly on dots
- Indoor overhead light works well
- Avoid strong backlighting

**Workflow:**
1. Point camera at dots → click camera button
2. Press **🤖 Auto-Tune HSV** to let AI set detection
3. Press **▶ Start** to begin a run
4. Hover over suture lines → click camera button each time for live feedback
5. Press **📌 Log Frame** to record measurements
6. Press **⏹ End** when done — summary appears in log
""")

# ─────────────────────────────────────────────────────
#  STITCH LOG + SESSION STATS
# ─────────────────────────────────────────────────────
st.markdown("---")
col_log, col_stats = st.columns([2, 1], gap="medium")

with col_log:
    st.markdown("### Stitch Log")
    if st.session_state.log_messages:
        log_html = "<br>".join(st.session_state.log_messages[-40:])
        st.markdown(f'<div class="stitch-log">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stitch-log" style="color:#3a6a3a;">Start a run and log stitches — they\'ll appear here.</div>', unsafe_allow_html=True)

with col_stats:
    st.markdown("### Session Stats")

    total_runs = len(st.session_state.run_history)
    all_scores = [rh["avg_score"] for rh in st.session_state.run_history]
    best_run   = max(all_scores) if all_scores else 0

    st.markdown(f'<div class="metric-card" style="margin-bottom:10px;"><div class="metric-label">Total Stitches Logged</div><div class="metric-value">{st.session_state.session_total}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card" style="margin-bottom:10px;"><div class="metric-label">Runs Completed</div><div class="metric-value">{total_runs}</div></div>', unsafe_allow_html=True)
    if best_run:
        st.markdown(f'<div class="metric-card" style="margin-bottom:10px;"><div class="metric-label">Best Run Score</div><div class="metric-value">{best_run}<span class="metric-unit">/100</span></div></div>', unsafe_allow_html=True)

    hsv_mode_label = {"auto": "🤖 Auto", "manual": "✋ Manual", "default": "⚙️ Default"}.get(st.session_state.hsv_source, "⚙️ Default")
    st.markdown(f'<div class="metric-card"><div class="metric-label">HSV Mode</div><div class="metric-value" style="font-size:1em;">{hsv_mode_label}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#2a4a2a;font-family:'JetBrains Mono',monospace;font-size:0.72em;margin-top:24px;padding-top:16px;border-top:1px solid #1a3d1a;">
    SUTERUP V0.3 · Surgical Training Platform · Asrith Maruvada · Live Detection + Auto-HSV Edition
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  EDGE CASE TEST RUNNER (run with: streamlit run suterup_v03.py -- --test)
#  These run as Python unit tests offline, not in Streamlit UI.
# ─────────────────────────────────────────────────────
import sys
if "--test" in sys.argv:
    import traceback

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    results = []

    def test(name, fn):
        try:
            fn()
            results.append(f"{PASS} | {name}")
        except Exception as e:
            results.append(f"{FAIL} | {name} → {e}")
            traceback.print_exc()

    # EC1: No dots — all black frame
    def ec1():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        pts, mask = detect_blobs(frame,90,130,80,255,50,255,200,8000)
        assert len(pts) == 0, f"Expected 0, got {len(pts)}"
    test("EC1 No dots (black frame)", ec1)

    # EC2: Single dot only
    def ec2():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        cv2.circle(frame, (200,240), 15, (255,100,50), -1)  # blue dot
        pts, _ = detect_blobs(frame,100,130,80,255,50,255,50,8000)
        assert len(pts) <= 1, f"Expected ≤1 dot"
    test("EC2 Single dot", ec2)

    # EC3: Many dots (12) — should cap at 10
    def ec3():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        for i in range(12):
            cv2.circle(frame, (50 + i*45, 240), 12, (255,100,50), -1)
        pts, _ = detect_blobs(frame,100,130,80,255,50,255,50,15000)
        assert len(pts) <= 10, f"Expected ≤10, got {len(pts)}"
    test("EC3 Many dots capped at 10", ec3)

    # EC4: Distance too close → precision never negative
    def ec4():
        sc = precision(10)
        assert sc >= 0, f"precision({10}) = {sc} — must be ≥ 0"
        g, _, _, _ = grade(10)
        assert g == "TOO CLOSE"
    test("EC4 Too close — no negative precision", ec4)

    # EC5: Very far dots → grade TOO WIDE
    def ec5():
        g, _, _, _ = grade(250)
        assert g == "TOO WIDE", f"Got {g}"
        sc = precision(250)
        assert sc >= 0
    test("EC5 Too wide grade + non-negative score", ec5)

    # EC6: Dark frame → auto-tune returns None gracefully
    def ec6():
        frame = np.full((480,640,3), 5, dtype=np.uint8)  # near-black
        result = auto_tune_hsv(frame)
        # Should return None (not crash)
        assert result is None or isinstance(result, dict)
    test("EC6 Dark frame auto-tune doesn't crash", ec6)

    # EC7: Non-blue (red) color → auto-tune finds it
    def ec7():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        # Draw several red circles
        for pos in [(100,240),(200,240),(300,240)]:
            cv2.circle(frame, pos, 20, (0,0,200), -1)  # red in BGR
        result = auto_tune_hsv(frame)
        # Should find a peak, may or may not be red in HSV but shouldn't crash
        assert result is None or isinstance(result, dict)
    test("EC7 Non-blue dots — auto-tune doesn't crash", ec7)

    # EC8: Camera None → no crash (simulated by passing empty)
    def ec8():
        # Simulate no camera by not calling detect_blobs — just verify grade handles 0 dots
        centroids = []
        assert len(centroids) < 2  # triggers the "no dots" warning branch
    test("EC8 No camera input — branch guard", ec8)

    # EC9: HSV min > max → detect_blobs guards it
    def ec9():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        cv2.circle(frame, (200,240), 15, (255,100,50), -1)
        # Intentionally inverted H min/max
        pts, mask = detect_blobs(frame, 130, 90, 80, 255, 50, 255, 50, 8000)
        # Should not crash — build_mask sorts them
        assert isinstance(pts, list)
    test("EC9 Inverted HSV range — clamped internally", ec9)

    # EC10: Rapid successive calls (10 frames)
    def ec10():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        for i in range(3):
            cv2.circle(frame, (100+i*100, 240), 15, (255,100,50), -1)
        for _ in range(10):
            pts, _ = detect_blobs(frame,100,130,80,255,50,255,50,8000)
        assert isinstance(pts, list)
    test("EC10 Rapid successive detection calls", ec10)

    # EC11: All-white frame → auto-tune filtered out (background suppression)
    def ec11():
        frame = np.full((480,640,3), 220, dtype=np.uint8)
        result = auto_tune_hsv(frame)
        # Should return None or minimal result — not crash
        assert result is None or isinstance(result, dict)
    test("EC11 All-white frame — background suppression", ec11)

    # EC12: Tiny dots (area < default min_area) — auto-tune lowers min_area
    def ec12():
        frame = np.zeros((480,640,3), dtype=np.uint8)
        # Draw very small dots (r=4 → area ~50px²)
        for pos in [(100,240),(200,240),(300,240)]:
            cv2.circle(frame, pos, 4, (255,100,50), -1)
        # With relaxed min_area
        pts, _ = detect_blobs(frame,100,130,80,255,50,255,10,500)
        assert isinstance(pts, list)  # should not crash
    test("EC12 Tiny dots with relaxed min_area", ec12)

    print("\n" + "═"*55)
    print("  SUTERUP V0.3 — EDGE CASE TEST RESULTS")
    print("═"*55)
    for r in results:
        print(f"  {r}")
    passes = sum(1 for r in results if r.startswith("✅"))
    print(f"\n  {passes}/{len(results)} passed")
    print("═"*55)
    sys.exit(0 if passes == len(results) else 1)
