"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SUTERUP V5.0 — Surgical Suture Training Platform                        ║
║  Streamlit · Live Camera · Auto-HSV AI · Real-time Feedback              ║
║  Programmer: Guneet Singh                                                ║
╚══════════════════════════════════════════════════════════════════════════╝

INSTALL (one-time):
    pip install streamlit opencv-python numpy pillow

RUN:
    streamlit run suterup_v5.py

Then open http://localhost:8501 in your browser.
Allow camera access when prompted by the browser.

PAGES (via sidebar):
    • Live Training   — camera feed + real-time AI analysis
    • Session History — logged stitches, charts, run summaries
    • Calibration     — manual HSV controls + diagnostic mask view
    • About           — tech guide, tips, scoring rubric

WHAT'S NEW IN V5 vs V4:
    - Migrated from Gradio → Streamlit (runs in any browser, no install needed for users)
    - Multi-page navigation via sidebar
    - Live camera via Streamlit's st.camera_input (snapshot-on-change, browser native)
    - Continuous "Live Mode" auto-refresh loop using st.rerun()
    - Auto-HSV with visual HSV mask overlay so users can SEE what AI detected
    - Calibration page shows the raw mask preview + sliders side-by-side
    - Session history page with bar charts (st.bar_chart) for score trends
    - Dark surgical UI via custom CSS injected through st.markdown
    - Error guards on every camera/CV operation
    - Temporal smoothing preserved from V4
    - All free, no API keys, no paid services
"""

import streamlit as st          # core framework — web UI, state, routing
import cv2                      # OpenCV — image processing, HSV masking, contours
import numpy as np              # numerical arrays — pixel math
import math                     # sqrt, pi — geometric calculations
import json                     # export session data to JSON
import time                     # timestamps, sleep
from datetime import datetime   # human-readable timestamps
from collections import deque   # fixed-length smoothing buffer
from PIL import Image           # convert between PIL and numpy for Streamlit camera
import io                       # byte buffers for image encoding

# ─────────────────────────────────────────────────────
#  PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suterup — Surgical Training",  # browser tab title
    page_icon="⚕",                             # favicon
    layout="wide",                             # full-width layout
    initial_sidebar_state="expanded",          # sidebar always open on load
)

# ─────────────────────────────────────────────────────
#  DARK SURGICAL CSS INJECTION
#  Streamlit renders everything inside an iframe; we inject
#  a <style> block via st.markdown(unsafe_allow_html=True)
#  to override default white theme without needing a config file.
# ─────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Import display font (free via Google Fonts) ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Global overrides ── */
html, body, [class*="css"] {
    background-color: #080e12 !important;
    color: #c9d8e4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0b1520 !important;
    border-right: 1px solid #1a3a55 !important;
}
[data-testid="stSidebar"] * { color: #7aaac8 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.85em !important; letter-spacing: 0.05em; }

/* ── Buttons ── */
.stButton > button {
    background: #0b1e2e !important;
    border: 1px solid #1e6ea0 !important;
    color: #5bc4f5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78em !important;
    border-radius: 3px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
    padding: 8px 16px !important;
}
.stButton > button:hover {
    background: #102840 !important;
    border-color: #5bc4f5 !important;
}
.stButton > button:disabled {
    border-color: #1a2a3a !important;
    color: #2a4a5a !important;
    cursor: not-allowed !important;
}

/* ── Danger button override (red class via markdown trick) ── */
.danger button {
    border-color: #a03030 !important;
    color: #f07070 !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { accent-color: #5bc4f5; }
.stSlider label { color: #5aaabf !important; font-size: 0.75em !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0b1a28 !important;
    border: 1px solid #1a3a55 !important;
    border-radius: 4px !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { color: #4a8aaa !important; font-size: 0.72em !important; letter-spacing: 0.08em !important; }
[data-testid="stMetricValue"] { color: #5bc4f5 !important; font-size: 1.4em !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.7em !important; }

/* ── Text inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #0b1a28 !important;
    border: 1px solid #1a3a55 !important;
    color: #c9d8e4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #0b1a28 !important;
    border: 1px solid #1a3a55 !important;
    color: #5bc4f5 !important;
    font-size: 0.82em !important;
}

/* ── Info / warning boxes ── */
.stAlert {
    background: #0b1a28 !important;
    border: 1px solid #1a3a55 !important;
    color: #7aaac8 !important;
}

/* ── Dividers ── */
hr { border-color: #1a3a55 !important; }

/* ── Camera widget ── */
[data-testid="stCameraInput"] {
    border: 1px solid #1a3a55 !important;
    border-radius: 4px !important;
}

/* ── Images (annotated output) ── */
[data-testid="stImage"] img {
    border: 1px solid #1a3a55 !important;
    border-radius: 4px !important;
}

/* ── Sidebar radio selected ── */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + span {
    color: #5bc4f5 !important;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.72em;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #2a6a8a;
    margin-bottom: 6px;
}

/* ── Status card ── */
.status-card {
    background: #0b1a28;
    border: 1px solid #1a3a55;
    border-radius: 4px;
    padding: 14px 16px;
    font-size: 0.82em;
    line-height: 1.9;
}

/* ── Score bar ── */
.score-bar-wrap {
    background: #0a1520;
    border-radius: 2px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.score-bar-fill {
    height: 6px;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* ── Log lines ── */
.log-line {
    font-size: 0.74em;
    color: #5a8aaa;
    border-bottom: 1px solid #0d1e2e;
    padding: 3px 0;
    line-height: 1.7;
}

/* ── Page header ── */
.page-header {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5em;
    color: #5bc4f5;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 0.73em;
    color: #2a5a7a;
    letter-spacing: 0.05em;
    margin-bottom: 20px;
}

/* ── Grade badge ── */
.grade-optimal { color: #00e09a !important; }
.grade-short   { color: #f5d05b !important; }
.grade-wide    { color: #f5d05b !important; }
.grade-danger  { color: #f07070 !important; }

/* ── Scrollable log box ── */
.log-scroll {
    max-height: 320px;
    overflow-y: auto;
    background: #080e12;
    border: 1px solid #1a3a55;
    border-radius: 4px;
    padding: 10px 12px;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)   # inject CSS before anything else renders

# ─────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
#  st.session_state persists across reruns (button clicks, camera updates)
#  but resets when the browser tab refreshes.
# ─────────────────────────────────────────────────────
def _init_state():
    """Populate st.session_state with defaults on first load only."""

    # HSV color detection parameters (hue 0-179, sat/val 0-255 in OpenCV)
    if "hsv" not in st.session_state:
        st.session_state.hsv = dict(h_min=90, h_max=130, s_min=80, s_max=255, v_min=50, v_max=255)

    # Blob area bounds — controls what size dot counts as a stitch marker
    if "area" not in st.session_state:
        st.session_state.area = dict(min_area=200, max_area=8000)

    # Which method set the HSV: "default" | "auto" | "manual"
    if "hsv_source" not in st.session_state:
        st.session_state.hsv_source = "default"

    # Whether a recording run is currently active
    if "run_active" not in st.session_state:
        st.session_state.run_active = False

    # Stitches logged in the current run
    if "run_stitches" not in st.session_state:
        st.session_state.run_stitches = []

    # Cumulative stitch count across all runs this session
    if "session_total" not in st.session_state:
        st.session_state.session_total = 0

    # Human-readable log lines shown on history page
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []

    # List of completed run summaries (dicts) for history page
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

    # Rolling centroid buffer for temporal smoothing (last N frames)
    if "centroid_history" not in st.session_state:
        st.session_state.centroid_history = deque(maxlen=6)

    # Last auto-tune result dict (for diagnostics display on calibration page)
    if "auto_hsv_result" not in st.session_state:
        st.session_state.auto_hsv_result = None

    # Live mode toggle: continuously rerun to process camera frames
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False

    # Store the last successfully processed annotated frame for display
    if "last_annotated" not in st.session_state:
        st.session_state.last_annotated = None

    # Store last computed centroids so logging can read them without re-processing
    if "last_centroids" not in st.session_state:
        st.session_state.last_centroids = []

    # Active page stored in state (matches sidebar radio)
    if "page" not in st.session_state:
        st.session_state.page = "Live Training"

_init_state()   # run on every rerun — safe because of the "not in" guards above

# ─────────────────────────────────────────────────────
#  MATH & GRADING HELPERS
# ─────────────────────────────────────────────────────
def seg_dist(p1, p2):
    """Euclidean pixel distance between two (x,y) centroids."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def precision_score(d):
    """
    Map a pixel distance d → score 0-100.

    Design rationale:
      - Optimal suture spacing in pixel terms is calibrated at 90px for a
        standard webcam view at ~30cm from the skin model.
      - Score degrades linearly: 1 point lost per 0.9px of deviation.
      - max(0, ...) prevents negative scores.
      - Users should aim for 80+ consistently.
    """
    return max(0, int(100 - abs(d - 90) / 0.9))

# Grade thresholds define clinical quality zones.
# Each tuple: (upper_px_limit, label, bgr_color, feedback_text)
# bgr_color is used in OpenCV drawing; rgb_hex is derived for CSS display.
GRADE_TABLE = [
    (40,  "TOO CLOSE", (224, 82,  82),  "Stitches too close — widen placement"),
    (60,  "SHORT",     (255, 215,  0),  "Slightly narrow — move dots apart"),
    (120, "OPTIMAL",   (  0, 224, 154), "Perfect — within ideal suture range"),
    (160, "WIDE",      (255, 215,  0),  "Slightly wide — bring dots closer"),
    (999, "TOO WIDE",  (224, 82,  82),  "Too wide — reduce stitch distance"),
]

def grade(d):
    """
    Look up the grade for a given pixel distance.
    Returns (label, bgr_color_tuple, feedback_string)
    """
    for threshold, label, color, feedback in GRADE_TABLE:
        if d < threshold:
            return label, color, feedback
    return "TOO WIDE", (224, 82, 82), "Too wide — reduce stitch distance"

def bgr_to_hex(bgr):
    """Convert OpenCV BGR tuple → CSS hex string for HTML rendering."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

def grade_css_class(label):
    """Map grade label → CSS class name for color styling in HTML blocks."""
    mapping = {
        "OPTIMAL":   "grade-optimal",
        "SHORT":     "grade-short",
        "WIDE":      "grade-wide",
        "TOO CLOSE": "grade-danger",
        "TOO WIDE":  "grade-danger",
    }
    return mapping.get(label, "grade-short")

# ─────────────────────────────────────────────────────
#  AUTO-HSV ENGINE (6-pass adaptive tuner)
# ─────────────────────────────────────────────────────
def auto_tune_hsv(frame_bgr):
    """
    Automatically determine optimal HSV parameters for the dominant
    colored object (suture dot) in the frame.

    Returns a dict of parameters or None if detection fails.

    Six passes explained:
    ──────────────────────────────────────────────────
    Pass 1  CLAHE normalisation — equalises lighting across frame so that
            shadowed or overexposed dots still have detectable hue.

    Pass 2  Candidate pixel extraction — only look at pixels with enough
            saturation and value to be a real colored dot (not grey/white skin).
            Adaptive floors prevent both too-strict and too-loose filters.

    Pass 3  Hue histogram + background suppression — hues that dominate >4%
            of the frame are likely skin or background, so zero them out.
            Then smooth the histogram to avoid noisy spikes.

    Pass 4  Peak finding + spread analysis — find the dominant remaining hue
            and measure how wide its cluster is. Add safety margin for the
            final h_min/h_max window.

    Pass 5  Blob validation with circularity filter — test the derived
            parameters against real contours. Dots are circular; hair/fabric
            edges are not. If no valid blobs found, relax the s/v floor.

    Pass 6  Adaptive area estimation from frame resolution — so the same
            code works on 480p, 720p, or 1080p cameras without manual tuning.
    ──────────────────────────────────────────────────
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None   # guard against empty/None frames

    # ── Pass 1: CLAHE on L channel of Lab colorspace ─────────────────────
    # CLAHE = Contrast Limited Adaptive Histogram Equalisation
    # Lab separates luminance (L) from color (a,b) so we can boost contrast
    # without shifting hue — critical for consistent dot detection under
    # variable clinic/training-room lighting.
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)                            # split channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # local equaliser
    lab_eq = cv2.merge([clahe.apply(l), a, b])          # replace L with equalised L
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)  # back to BGR for HSV conversion

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)    # convert to HSV working space
    h_ch = hsv[:, :, 0]   # Hue channel   (0-179 in OpenCV, represents 0-360°)
    s_ch = hsv[:, :, 1]   # Saturation channel (0-255)
    v_ch = hsv[:, :, 2]   # Value channel      (0-255)

    # ── Pass 2: Adaptive candidate thresholds ────────────────────────────
    mean_v = float(np.mean(v_ch))   # mean brightness across frame
    mean_s = float(np.mean(s_ch))   # mean saturation across frame

    # Set floors relative to mean — in a dim room, mean_v is low so floor is low
    # This prevents rejecting real dots just because the environment is dark
    v_floor = max(15, int(mean_v * 0.22))    # require at least 22% of mean brightness
    s_floor = max(25, int(mean_s * 0.28))    # require at least 28% of mean saturation

    # Boolean mask: True where pixel might be a colored dot
    candidate_mask = (s_ch > s_floor) & (v_ch > v_floor)

    if candidate_mask.sum() < 80:
        # Fewer than 80 candidate pixels → frame is too dark or featureless
        return None

    candidate_hues = h_ch[candidate_mask]   # extract hue values of candidates only

    # ── Pass 3: Histogram + background suppression ───────────────────────
    # 180-bin histogram (one per OpenCV hue degree)
    hist, _ = np.histogram(candidate_hues, bins=180, range=(0, 179))

    # Box-smooth with 9-wide kernel to reduce single-bin spikes
    kernel = np.ones(9) / 9
    hist_sm = np.convolve(hist.astype(float), kernel, mode='same')

    # Suppress background hues: any hue covering >4% of the entire frame
    # is probably skin tone, surgical gown, or room background — not a dot
    total_px = frame_bgr.shape[0] * frame_bgr.shape[1]
    hist_sm[hist_sm > total_px * 0.04] = 0   # zero-out dominant background hues

    if hist_sm.max() < 30:
        return None   # no meaningful colored peak found after suppression

    # ── Pass 4: Peak + spread ────────────────────────────────────────────
    peak_hue = int(np.argmax(hist_sm))   # hue degree with highest count
    peak_val = hist_sm[peak_hue]
    thresh20 = peak_val * 0.20           # spread threshold: 20% of peak height

    # Measure how wide the hue cluster is around the peak
    spread = 0
    for delta in range(1, 35):
        left  = hist_sm[(peak_hue - delta) % 180] > thresh20   # wrap around 0/179
        right = hist_sm[(peak_hue + delta) % 180] > thresh20
        if left or right:
            spread = delta   # extend spread as long as cluster continues
        else:
            break            # stop when histogram drops below 20% of peak

    # Add 6px safety margin so border pixels aren't cut off
    tol   = max(14, spread + 6)
    h_min = max(0,   peak_hue - tol)
    h_max = min(179, peak_hue + tol)

    # Ensure minimum window width of 16° so tiny clusters still detect
    if (h_max - h_min) < 16:
        c = (h_min + h_max) // 2
        h_min, h_max = max(0, c - 8), min(179, c + 8)

    # Derive saturation/value bounds from pixels near the dominant hue
    near_peak = candidate_mask & (np.abs(h_ch.astype(int) - peak_hue) < tol)
    sat_near  = s_ch[near_peak]
    val_near  = v_ch[near_peak]
    # Use 8th percentile as floor so 92% of real dot pixels pass through
    s_min = max(25, int(np.percentile(sat_near, 8))) if len(sat_near) > 0 else s_floor
    v_min = max(15, int(np.percentile(val_near, 8))) if len(val_near) > 0 else v_floor

    # ── Pass 5: Validate with circularity check ──────────────────────────
    img_area = total_px
    # Initial area bounds before adaptive sizing
    min_area = max(40,  int(img_area * 0.00025))   # 0.025% of frame = smallest dot
    max_area = min(int(img_area * 0.05), 22000)    # 5% of frame = largest dot

    # Build a test mask and count valid (circular) blobs
    test_mask = _build_mask_internal(enhanced, h_min, h_max, s_min, 255, v_min, 255)
    cnts, _   = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                # Circularity = 4π·area / perimeter² (=1.0 for perfect circle)
                # Reject non-circular shapes like hair strands, fabric edges, creases
                circularity = 4 * math.pi * area / (perimeter ** 2)
                if circularity > 0.25:    # 0.25 = roughly blob-like (squares ~0.79)
                    valid_blobs += 1

    # If 0 valid blobs, relax s/v floors and try less strict detection
    if valid_blobs == 0:
        s_min = max(20, s_min - 20)   # widen saturation floor
        v_min = max(15, v_min - 20)   # widen brightness floor

    # ── Pass 6: Adaptive area bounds from frame resolution ───────────────
    fh = frame_bgr.shape[0]           # frame height in pixels
    # Assume suture dots occupy 0.8%-6% of frame height as radius
    r_min = max(3, int(fh * 0.008))   # minimum dot radius in pixels
    r_max = max(r_min + 5, int(fh * 0.06))   # maximum dot radius in pixels
    # Convert radius → area bounds (πr²) with 30% tolerance margins
    min_area = max(40, int(math.pi * r_min**2 * 0.7))
    max_area = min(22000, int(math.pi * r_max**2 * 1.3))

    return {
        "h_min": h_min, "h_max": h_max,
        "s_min": s_min, "s_max": 255,
        "v_min": v_min, "v_max": 255,
        "min_area": min_area, "max_area": max_area,
        "peak_hue": peak_hue,
        "valid_blobs": valid_blobs,
        "spread": spread,
        "tol": tol,
    }

# ─────────────────────────────────────────────────────
#  MASK BUILDING (internal + public wrappers)
# ─────────────────────────────────────────────────────
def _build_mask_internal(frame_bgr, h_min, h_max, s_min, s_max, v_min, v_max):
    """
    Convert BGR frame to HSV, apply color range mask, then morphologically
    clean it up to remove noise and fill small holes in blobs.

    Morphological operations:
      OPEN  (erode then dilate) → removes speckle noise smaller than kernel
      CLOSE (dilate then erode) → fills small gaps/holes inside blobs
    Both use a 5x5 elliptical kernel matching the circular shape of dots.
    """
    # Clip all parameters to valid OpenCV ranges
    h_min, h_max = sorted([int(np.clip(h_min, 0, 179)), int(np.clip(h_max, 0, 179))])
    s_min, s_max = sorted([int(np.clip(s_min, 0, 255)), int(np.clip(s_max, 0, 255))])
    v_min, v_max = sorted([int(np.clip(v_min, 0, 255)), int(np.clip(v_max, 0, 255))])

    # Apply CLAHE for consistent detection under variable lighting (same as auto-tune)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2  = cv2.merge([clahe.apply(l), a, b])
    enh   = cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)

    hsv   = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)   # lower HSV bound
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)   # upper HSV bound
    mask  = cv2.inRange(hsv, lower, upper)                     # binary mask

    # Elliptical 5x5 structuring element — matches dot shape better than square
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)   # noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)   # hole filling
    return mask

def build_mask(frame_bgr):
    """
    Public wrapper — reads HSV parameters from session state.
    Returns binary mask (numpy uint8 array, 0 or 255).
    """
    h = st.session_state.hsv
    a = st.session_state.area
    return _build_mask_internal(
        frame_bgr,
        h["h_min"], h["h_max"],
        h["s_min"], h["s_max"],
        h["v_min"], h["v_max"],
    )

# ─────────────────────────────────────────────────────
#  BLOB DETECTION
# ─────────────────────────────────────────────────────
def detect_blobs(frame_bgr):
    """
    Find suture dot centroids in the frame.

    Algorithm:
      1. Build HSV color mask (isolates dot color).
      2. Find external contours in the mask.
      3. Filter by area bounds (reject too-small noise and too-large patches).
      4. Filter by circularity ≥ 0.20 (reject non-dot shapes).
      5. Compute centroid via image moments.
      6. Sort by x then nearest-neighbour chain for consistent left-to-right ordering.

    Returns:
      centroids — list of (cx, cy) tuples, up to 10
      mask      — the raw binary mask (for display on calibration page)
    """
    h_p = st.session_state.hsv
    a_p = st.session_state.area

    min_a = max(10, int(a_p["min_area"]))    # absolute minimum blob area in px²
    max_a = max(min_a + 100, int(a_p["max_area"]))   # absolute maximum

    mask = _build_mask_internal(
        frame_bgr,
        h_p["h_min"], h_p["h_max"],
        h_p["s_min"], h_p["s_max"],
        h_p["v_min"], h_p["v_max"],
    )

    # Find all external contours — CHAIN_APPROX_SIMPLE compresses horizontal/vertical
    # runs into endpoints to save memory (fine for centroid calculation)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []    # will hold (cx, cy, area) for each valid blob
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_a <= area <= max_a):
            continue   # area filter — skip too-small noise or too-large patches

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue   # degenerate contour — skip

        # Circularity = 4π·A / P² — equals 1.0 for a perfect circle
        # Hair strands: ~0.01-0.05, fabric edges: ~0.05-0.15, dots: ≥0.25
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < 0.20:
            continue   # reject non-circular blobs

        # Image moments: m10/m00 = centroid x, m01/m00 = centroid y
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])   # x centroid
            cy = int(M["m01"] / M["m00"])   # y centroid
            pts.append((cx, cy, area))

    if len(pts) < 2:
        # Return whatever we have (0 or 1 points) — caller handles this case
        return [(p[0], p[1]) for p in pts[:1]], mask

    # Sort all blobs left-to-right as starting point
    pts = sorted(pts, key=lambda p: p[0])

    # Nearest-neighbour chain: ensures P1→P2→P3... follows physical stitch order
    # rather than arbitrary left-to-right, which can skip around if dots misalign
    ordered = [pts.pop(0)]
    while pts:
        last = ordered[-1]
        # Find the closest remaining point to the current chain tail
        idx = min(range(len(pts)),
                  key=lambda i: (pts[i][0] - last[0])**2 + (pts[i][1] - last[1])**2)
        ordered.append(pts.pop(idx))

    return [(p[0], p[1]) for p in ordered[:10]], mask   # cap at 10 dots

# ─────────────────────────────────────────────────────
#  TEMPORAL SMOOTHING
# ─────────────────────────────────────────────────────
def smooth_centroids(new_centroids):
    """
    Average centroids across the last N frames to reduce jitter from
    minor camera shake or single-frame detection noise.

    Only smooths if dot count is stable (same count in all buffered frames),
    because averaging a 3-dot frame with a 2-dot frame produces garbage.

    Modifies st.session_state.centroid_history in-place.
    Returns smoothed centroid list.
    """
    history = st.session_state.centroid_history

    if not new_centroids:
        history.clear()   # reset history when dots disappear
        return new_centroids

    history.append(new_centroids)   # add new frame to rolling buffer

    # Check count stability across all buffered frames
    counts = [len(f) for f in history]
    if len(set(counts)) != 1:
        return new_centroids   # inconsistent count — use raw (no averaging)

    n = len(history)   # number of frames in buffer
    k = len(new_centroids)   # number of dots per frame
    smoothed = []
    for i in range(k):
        # Average the i-th centroid across all buffered frames
        avg_x = int(sum(history[j][i][0] for j in range(n)) / n)
        avg_y = int(sum(history[j][i][1] for j in range(n)) / n)
        smoothed.append((avg_x, avg_y))
    return smoothed

# ─────────────────────────────────────────────────────
#  FRAME ANNOTATION (OpenCV drawing)
# ─────────────────────────────────────────────────────
# Color palette for dot circles — each dot gets a distinct color
DOT_COLORS_BGR = [
    (154, 224,   0), (255, 140,  66), (255, 200,  77),
    (255, 120, 200), (255, 219,  77), (130, 255, 140),
    ( 60, 180, 255), (255,  80,  60), (180,  80, 220),
    (180, 255,  80),
]

def draw_dashed_line(frame, p1, p2, color, thickness=2):
    """
    Draw a dashed line between two points.
    Alternates between 10px drawn segments and 5px gaps.
    Used to visually link suture dot pairs on the annotated output.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max(1, math.sqrt(dx**2 + dy**2))   # total line length
    ux, uy = dx / length, dy / length             # unit direction vector
    pos, on = 0.0, True                           # position along line, draw flag

    while pos < length:
        end = min(pos + (10 if on else 5), length)   # segment end position
        if on:
            # Compute actual pixel positions along the direction vector
            a = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
            b = (int(p1[0] + ux * end), int(p1[1] + uy * end))
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
        pos, on = end, not on   # advance and toggle draw/skip

def annotate_frame(frame_bgr, centroids, hsv_source="auto"):
    """
    Draw all visual overlays onto a copy of the frame:
      - Dashed lines between consecutive dot pairs with distance labels
      - Dot circles with labels (P1, P2, ...)
      - HUD panel top-left: dot count, avg score, distance, HSV mode
      - Status bar bottom: RUN/STANDBY, session stitch count
      - If <2 dots: "NO DOTS DETECTED" overlay message

    Returns annotated BGR frame.
    """
    out = frame_bgr.copy()   # never modify the original frame
    h, w = out.shape[:2]

    if len(centroids) >= 2:
        # Compute distances and scores for all consecutive pairs
        dists  = [seg_dist(centroids[i], centroids[i + 1]) for i in range(len(centroids) - 1)]
        scores = [precision_score(d) for d in dists]
        avg_sc = int(sum(scores) / len(scores))   # average score across all pairs
        avg_d  = sum(dists) / len(dists)           # average distance across all pairs

        # ── Draw segment lines and labels ──
        for i in range(len(centroids) - 1):
            p1, p2 = centroids[i], centroids[i + 1]
            d = dists[i]
            label, color, _ = grade(d)
            sc = scores[i]

            draw_dashed_line(out, p1, p2, color, 2)

            # Midpoint label position
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            # Draw two text labels: pixel distance on top, grade+score below
            for txt, dy_off in [(f"{d:.0f}px", -18), (f"{label}  {sc}/100", 0)]:
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                # Dark background rectangle behind text for readability
                cv2.rectangle(out,
                              (mid[0] - 4, mid[1] + dy_off - th - 2),
                              (mid[0] + tw + 4, mid[1] + dy_off + 2),
                              (8, 14, 20), -1)   # very dark blue-black background
                cv2.putText(out, txt, (mid[0], mid[1] + dy_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # ── Draw dot circles ──
        for i, pt in enumerate(centroids):
            c = DOT_COLORS_BGR[i % len(DOT_COLORS_BGR)]
            cv2.circle(out, pt, 18, c, 1, cv2.LINE_AA)    # outer ring
            cv2.circle(out, pt, 10, c, -1, cv2.LINE_AA)   # filled inner
            cv2.circle(out, pt,  3, (245, 255, 250), -1, cv2.LINE_AA)   # bright center dot
            cv2.putText(out, f"P{i + 1}", (pt[0] + 20, pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, c, 1, cv2.LINE_AA)

        # ── HUD top-left ──
        run_state = st.session_state.run_active
        hud = [
            (f"DOTS: {len(centroids)}", (180, 220, 200)),
            (f"AVG SCORE: {avg_sc}/100", (0, 224, 154) if avg_sc >= 70 else (255, 215, 0) if avg_sc >= 40 else (224, 82, 82)),
            (f"AVG DIST: {avg_d:.0f}px", (180, 220, 200)),
            (f"MODE: {hsv_source.upper()}", (0, 180, 140)),
        ]
        for j, (txt, col) in enumerate(hud):
            y = 12 + j * 22 + 18
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(out, (8, y - th - 3), (12 + tw + 4, y + 3), (6, 12, 18), -1)
            cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

        # ── Status bar ──
        rec_txt = "● REC" if run_state else "○ STANDBY"
        rec_col = (0, 80, 220) if run_state else (60, 90, 70)
        cv2.rectangle(out, (0, h - 28), (w, h), (6, 12, 18), -1)
        cv2.putText(out,
                    f"SUTERUP V5  {rec_txt}  SESSION: {st.session_state.session_total} stitches",
                    (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rec_col, 1, cv2.LINE_AA)

    else:
        # ── No-dots overlay ──
        msg = ("NO DOTS DETECTED — run Auto-Tune or adjust HSV"
               if len(centroids) == 0 else "1 DOT FOUND — need at least 2")
        overlay_y = h // 2
        cv2.rectangle(out, (0, overlay_y - 24), (w, overlay_y + 12), (8, 14, 24), -1)
        cv2.putText(out, msg, (12, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 130, 255), 1, cv2.LINE_AA)

    return out

# ─────────────────────────────────────────────────────
#  FRAME INGESTION: camera_input → BGR numpy
# ─────────────────────────────────────────────────────
def camera_to_bgr(camera_file):
    """
    Convert Streamlit's st.camera_input output (uploaded file object, JPEG bytes)
    into an OpenCV BGR numpy array for processing.

    Steps:
      1. Read raw bytes from the file-like object.
      2. Decode JPEG bytes into a numpy array via cv2.imdecode.
      3. The resulting array is already BGR (OpenCV default).

    Returns numpy array shape (H, W, 3) or None on failure.
    """
    if camera_file is None:
        return None
    try:
        # Read bytes from the file buffer
        raw_bytes = camera_file.read()
        # Convert raw bytes to 1D numpy array then decode as image
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # decode to BGR
        return bgr
    except Exception as e:
        # Log to Streamlit console but don't crash the app
        st.warning(f"Camera read error: {e}")
        return None

def bgr_to_rgb_pil(bgr):
    """
    Convert BGR numpy array → RGB PIL Image for st.image() display.
    Streamlit's st.image() expects RGB, not BGR.
    """
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # swap R and B channels
    return Image.fromarray(rgb)                   # wrap in PIL Image

# ─────────────────────────────────────────────────────
#  PROCESS PIPELINE (called on every camera frame)
# ─────────────────────────────────────────────────────
def process_camera_frame(camera_file):
    """
    Full pipeline: camera file → annotated image + centroid data.

    1. Convert camera file to BGR.
    2. Detect blobs → raw centroids.
    3. Apply temporal smoothing.
    4. Annotate frame with overlays.
    5. Store results in session state for the log button.
    6. Convert annotated BGR → RGB PIL for display.

    Returns (pil_image, centroids) or (None, []) on failure.
    """
    bgr = camera_to_bgr(camera_file)
    if bgr is None:
        return None, []

    try:
        centroids_raw, _mask = detect_blobs(bgr)
        centroids = smooth_centroids(centroids_raw)
        annotated_bgr = annotate_frame(bgr, centroids, st.session_state.hsv_source)
        st.session_state.last_annotated = annotated_bgr   # cache for potential re-display
        st.session_state.last_centroids = centroids        # cache for log button
        return bgr_to_rgb_pil(annotated_bgr), centroids
    except Exception as e:
        st.error(f"Processing error: {e}")
        return bgr_to_rgb_pil(bgr), []   # return raw frame on error, don't crash

# ─────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="padding:16px 0 20px 0;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5em;font-weight:700;
                    color:#5bc4f5;letter-spacing:-0.03em;">⚕ SUTERUP</div>
        <div style="font-size:0.65em;color:#1e4a6a;letter-spacing:0.12em;">V5.0 · SURGICAL TRAINING</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Multi-page navigation
    page = st.radio(
        "NAVIGATE",
        ["Live Training", "Session History", "Calibration", "About"],
        index=["Live Training", "Session History", "Calibration", "About"].index(
            st.session_state.page
        ),
        label_visibility="visible",
    )
    st.session_state.page = page   # persist selected page

    st.markdown("---")

    # Quick session stats in sidebar
    st.markdown("""<div class="section-label">Session</div>""", unsafe_allow_html=True)
    st.metric("Total Stitches", st.session_state.session_total)
    st.metric("Runs Completed", len(st.session_state.run_history))

    run_color = "#00e09a" if st.session_state.run_active else "#2a4a5a"
    run_label = "● RECORDING" if st.session_state.run_active else "○ STANDBY"
    st.markdown(
        f"<div style='font-size:0.72em;color:{run_color};letter-spacing:0.1em;margin-top:8px;'>{run_label}</div>",
        unsafe_allow_html=True,
    )

    # Reset session button in sidebar
    st.markdown("---")
    if st.button("🔄 Reset Session", key="sidebar_reset"):
        # Clear all session-specific state
        st.session_state.run_active    = False
        st.session_state.run_stitches  = []
        st.session_state.session_total = 0
        st.session_state.log_lines     = ["🔄 Session reset"]
        st.session_state.run_history   = []
        st.session_state.centroid_history.clear()
        st.session_state.last_annotated = None
        st.session_state.last_centroids  = []
        st.rerun()   # re-render the full page with cleared state

    st.markdown("""
    <div style="position:absolute;bottom:20px;left:20px;font-size:0.65em;color:#0f2535;">
        Asrith Maruvada · Beta
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  PAGE: LIVE TRAINING
# ─────────────────────────────────────────────────────
if page == "Live Training":

    st.markdown("""
    <div class="page-header">Live Training Feed</div>
    <div class="page-sub">REAL-TIME SUTURE ANALYSIS · PLACE DOTS IN FRAME · AUTO-AI DETECTION</div>
    """, unsafe_allow_html=True)

    # ── Layout: camera left | controls right ──
    col_cam, col_ctrl = st.columns([2.2, 1], gap="medium")

    with col_cam:
        # Camera input widget — triggers on each new photo taken
        # In Streamlit, st.camera_input shows a live viewfinder; user clicks
        # the shutter to capture. For continuous mode we auto-rerun.
        camera_file = st.camera_input(
            label="Point at suture dots and capture",
            label_visibility="collapsed",
            key="camera_feed",
        )

        # Process frame whenever a new capture arrives
        if camera_file is not None:
            pil_annotated, centroids = process_camera_frame(camera_file)
            if pil_annotated:
                st.image(pil_annotated, caption="Annotated Output", use_container_width=True)
        elif st.session_state.last_annotated is not None:
            # Show last frame if no new capture yet (prevents blank screen on rerun)
            st.image(
                bgr_to_rgb_pil(st.session_state.last_annotated),
                caption="Last Frame",
                use_container_width=True,
            )

        # ── Live feedback status card ──
        centroids = st.session_state.last_centroids
        if len(centroids) >= 2:
            dists  = [seg_dist(centroids[i], centroids[i + 1]) for i in range(len(centroids) - 1)]
            scores = [precision_score(d) for d in dists]
            avg_sc = int(sum(scores) / len(scores))
            avg_d  = sum(dists) / len(dists)
            last_d = dists[-1]
            label, color_bgr, feedback = grade(last_d)
            hex_col = bgr_to_hex(color_bgr)
            css_cls = grade_css_class(label)

            # Score bar fill color matches grade
            bar_color = hex_col

            # Build segment table rows
            rows_html = ""
            for i, (d, sc) in enumerate(zip(dists, scores)):
                gl, gc, _ = grade(d)
                hc = bgr_to_hex(gc)
                rows_html += (
                    f"<tr>"
                    f"<td style='color:#3a6a8a;padding:3px 8px;'>P{i+1}→P{i+2}</td>"
                    f"<td style='color:{hc};padding:3px 8px;'>{d:.1f}px</td>"
                    f"<td style='color:{hc};padding:3px 8px;'>{gl}</td>"
                    f"<td style='color:{hc};padding:3px 8px;'>{sc}/100</td>"
                    f"</tr>"
                )

            st.markdown(f"""
            <div class="status-card" style="margin-top:12px;">
                <div style="font-size:1.1em;font-weight:700;color:{hex_col};margin-bottom:4px;">
                    {label} &nbsp;·&nbsp; {avg_sc}/100
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-fill"
                         style="width:{avg_sc}%;background:{bar_color};"></div>
                </div>
                <div style="color:#4a7a9a;font-size:0.78em;margin:8px 0 10px 0;">
                    💬 {feedback}
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:0.76em;">
                    <tr style="color:#1e4a6a;border-bottom:1px solid #0e2030;">
                        <td style="padding:3px 8px;">SEG</td>
                        <td style="padding:3px 8px;">DIST</td>
                        <td style="padding:3px 8px;">GRADE</td>
                        <td style="padding:3px 8px;">SCORE</td>
                    </tr>
                    {rows_html}
                </table>
                <div style="margin-top:10px;color:#2a5a7a;font-size:0.73em;">
                    AVG DIST: <span style="color:#5bc4f5;">{avg_d:.1f}px</span>
                    &nbsp;·&nbsp;
                    DOTS: <span style="color:#5bc4f5;">{len(centroids)}</span>
                    &nbsp;·&nbsp;
                    SESSION TOTAL: <span style="color:#5bc4f5;">{st.session_state.session_total}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif len(centroids) == 1:
            st.markdown("""
            <div class="status-card" style="color:#f5d05b;margin-top:12px;">
                ⚠ 1 dot detected — need at least 2 to measure distance
            </div>
            """, unsafe_allow_html=True)
        elif camera_file is not None:
            st.markdown("""
            <div class="status-card" style="color:#4a8aaa;margin-top:12px;">
                ⚠ No dots detected — try Auto-Tune HSV or adjust Calibration
            </div>
            """, unsafe_allow_html=True)

    with col_ctrl:
        # ── Session control panel ──
        st.markdown('<div class="section-label">Session Controls</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            # Start button: disabled if already recording
            if st.button("▶ Start Run",
                         disabled=st.session_state.run_active,
                         key="btn_start",
                         use_container_width=True):
                st.session_state.run_active   = True
                st.session_state.run_stitches = []
                st.session_state.centroid_history.clear()
                ts = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_lines.append(f"🟢 [{ts}] Run started")
                st.rerun()

        with c2:
            # End button: disabled if not recording
            if st.button("⏹ End Run",
                         disabled=not st.session_state.run_active,
                         key="btn_end",
                         use_container_width=True):
                st.session_state.run_active = False
                run_s = st.session_state.run_stitches
                ts = datetime.now().strftime("%H:%M:%S")
                if run_s:
                    s_scores = [s["sc"] for s in run_s]
                    s_dists  = [s["d"]  for s in run_s]
                    optimal  = sum(1 for s in run_s if 60 <= s["d"] <= 120)
                    avg_run  = int(sum(s_scores) / max(1, len(s_scores)))
                    # Archive completed run summary to history
                    st.session_state.run_history.append({
                        "ts": ts,
                        "stitches": len(run_s),
                        "avg_score": avg_run,
                        "optimal": optimal,
                        "best":  max(s_scores),
                        "worst": min(s_scores),
                        "avg_dist": sum(s_dists) / max(1, len(s_dists)),
                    })
                    st.session_state.log_lines += [
                        "━" * 28,
                        f"🏁 [{ts}] RUN COMPLETE",
                        f"   Stitches : {len(run_s)}",
                        f"   Avg score: {avg_run}/100",
                        f"   Optimal  : {optimal}/{len(run_s)}",
                        "━" * 28,
                    ]
                st.session_state.run_stitches = []
                st.rerun()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Log current frame button
        if st.button("📌 Log Current Frame",
                     disabled=not st.session_state.run_active,
                     key="btn_log",
                     use_container_width=True):
            cents = st.session_state.last_centroids
            if len(cents) < 2:
                st.session_state.log_lines.append("⚠ Log skipped — fewer than 2 dots")
            else:
                ts = datetime.now().strftime("%H:%M:%S")
                count = 0
                for i in range(len(cents) - 1):
                    d  = seg_dist(cents[i], cents[i + 1])
                    sc = precision_score(d)
                    gl, _, _ = grade(d)
                    st.session_state.run_stitches.append({"d": d, "sc": sc})
                    st.session_state.session_total += 1
                    count += 1
                    st.session_state.log_lines.append(
                        f"📌 [{ts}] #{st.session_state.session_total} P{i+1}→P{i+2} "
                        f"| {d:.1f}px | {gl} | {sc}/100"
                    )
                st.session_state.log_lines.append(f"✅ {count} stitch(es) saved")
            st.rerun()

        st.markdown("---")

        # ── Auto-HSV section ──
        st.markdown('<div class="section-label">AI Detection</div>', unsafe_allow_html=True)

        hsv_mode_map = {"default": "Default", "auto": "Auto-Tuned ✓", "manual": "Manual Override"}
        st.markdown(
            f"<div style='font-size:0.72em;color:#2a6a8a;margin-bottom:8px;'>"
            f"Mode: <span style='color:#5bc4f5;'>{hsv_mode_map.get(st.session_state.hsv_source,'?')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("🤖 Auto-Tune HSV", key="btn_autotune", use_container_width=True):
            cam = st.session_state.get("camera_feed")   # last captured frame
            bgr = camera_to_bgr(cam)
            if bgr is None:
                st.warning("Capture a frame first — point camera at dots then click Auto-Tune")
            else:
                with st.spinner("Running 6-pass Auto-HSV..."):
                    result = auto_tune_hsv(bgr)
                if result:
                    st.session_state.hsv = {k: result[k] for k in
                                            ("h_min","h_max","s_min","s_max","v_min","v_max")}
                    st.session_state.area = {"min_area": result["min_area"],
                                             "max_area": result["max_area"]}
                    st.session_state.hsv_source   = "auto"
                    st.session_state.auto_hsv_result = result
                    st.session_state.centroid_history.clear()
                    st.success(
                        f"✅ Auto-HSV complete — hue peak H={result['peak_hue']}° · "
                        f"{result['valid_blobs']} dots validated"
                    )
                else:
                    st.error("Auto-tune failed — improve lighting or move closer to dots")

        # Show auto-tune diagnostic if available
        if st.session_state.auto_hsv_result:
            r = st.session_state.auto_hsv_result
            with st.expander("📊 Auto-tune Diagnostic", expanded=False):
                st.markdown(f"""
                <div style="font-size:0.74em;color:#5aaabf;line-height:2;">
                  Peak hue: <span style="color:#5bc4f5;">{r['peak_hue']}°</span><br>
                  Hue spread: ±{r['spread']}px · window: {r['h_min']}–{r['h_max']}<br>
                  S min: {r['s_min']} · V min: {r['v_min']}<br>
                  Blob area: {r['min_area']}–{r['max_area']} px²<br>
                  Valid blobs found: <span style="color:#5bc4f5;">{r['valid_blobs']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Setup tips ──
        st.markdown('<div class="section-label">Setup Tips</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72em;color:#2a5a7a;line-height:2;">
          • Use <span style="color:#7aaac8;">blue, red, or green stickers</span><br>
          • Even lighting — no shadows on dots<br>
          • Space dots <span style="color:#7aaac8;">1–3 cm</span> apart<br>
          • Click <span style="color:#5bc4f5;">Auto-Tune HSV</span> first<br>
          • OPTIMAL = 60–120px between dots<br>
          • Capture frame → auto processes instantly
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  PAGE: SESSION HISTORY
# ─────────────────────────────────────────────────────
elif page == "Session History":

    st.markdown("""
    <div class="page-header">Session History</div>
    <div class="page-sub">LOGGED STITCHES · RUN SUMMARIES · SCORE TRENDS</div>
    """, unsafe_allow_html=True)

    if not st.session_state.run_history and not st.session_state.log_lines:
        st.markdown("""
        <div class="status-card" style="color:#2a5a7a;text-align:center;padding:40px;">
            No data yet — complete a run on the Live Training page.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Summary metrics ──
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Stitches", st.session_state.session_total)
        with col_m2:
            st.metric("Runs Completed", len(st.session_state.run_history))
        with col_m3:
            if st.session_state.run_history:
                avg_scores = [r["avg_score"] for r in st.session_state.run_history]
                st.metric("Avg Score", f"{int(sum(avg_scores)/len(avg_scores))}/100")
            else:
                st.metric("Avg Score", "—")
        with col_m4:
            if st.session_state.run_history:
                opt_rates = [r["optimal"] / max(1, r["stitches"]) * 100
                             for r in st.session_state.run_history]
                st.metric("Optimal Rate", f"{int(sum(opt_rates)/len(opt_rates))}%")
            else:
                st.metric("Optimal Rate", "—")

        st.markdown("---")

        # ── Score trend chart ──
        if len(st.session_state.run_history) >= 1:
            st.markdown('<div class="section-label">Score Trend by Run</div>', unsafe_allow_html=True)
            chart_data = {
                "Run": [f"Run {i+1}" for i, _ in enumerate(st.session_state.run_history)],
                "Avg Score": [r["avg_score"] for r in st.session_state.run_history],
            }
            # st.bar_chart expects a dict or dataframe; index = x axis
            import pandas as pd
            df = pd.DataFrame(chart_data).set_index("Run")
            st.bar_chart(df, color="#5bc4f5", height=220)

        st.markdown("---")

        # ── Run history table ──
        if st.session_state.run_history:
            st.markdown('<div class="section-label">Run Summaries</div>', unsafe_allow_html=True)
            for i, r in enumerate(reversed(st.session_state.run_history)):
                grade_color = "#00e09a" if r["avg_score"] >= 70 else "#f5d05b" if r["avg_score"] >= 40 else "#f07070"
                opt_pct = int(r["optimal"] / max(1, r["stitches"]) * 100)
                with st.expander(f"Run {len(st.session_state.run_history) - i} · {r['ts']} · Avg {r['avg_score']}/100"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Stitches", r["stitches"])
                    c2.metric("Optimal", f"{r['optimal']}/{r['stitches']} ({opt_pct}%)")
                    c3.metric("Avg Dist", f"{r['avg_dist']:.1f}px")
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Best Score", f"{r['best']}/100")
                    cc2.metric("Worst Score", f"{r['worst']}/100")

        st.markdown("---")

        # ── Stitch event log ──
        st.markdown('<div class="section-label">Event Log</div>', unsafe_allow_html=True)
        if st.session_state.log_lines:
            # Build HTML scrollable log
            log_html = "<div class='log-scroll'>"
            for line in reversed(st.session_state.log_lines[-60:]):
                log_html += f"<div class='log-line'>{line}</div>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

        st.markdown("---")

        # ── Export ──
        st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
        if st.button("⬇ Export Session JSON"):
            export_data = {
                "export_time": datetime.now().isoformat(),
                "session_total": st.session_state.session_total,
                "run_history": st.session_state.run_history,
                "log": st.session_state.log_lines,
            }
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"suterup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

# ─────────────────────────────────────────────────────
#  PAGE: CALIBRATION
# ─────────────────────────────────────────────────────
elif page == "Calibration":

    st.markdown("""
    <div class="page-header">Calibration</div>
    <div class="page-sub">MANUAL HSV CONTROL · MASK DIAGNOSTIC · AREA TUNING</div>
    """, unsafe_allow_html=True)

    col_sliders, col_preview = st.columns([1, 1.5], gap="medium")

    with col_sliders:
        st.markdown('<div class="section-label">HSV Range</div>', unsafe_allow_html=True)

        h = st.session_state.hsv   # current HSV dict

        # Each slider maps to an HSV channel bound.
        # OpenCV uses H: 0-179 (half of 360° wheel), S/V: 0-255.
        new_h_min = st.slider("Hue Min (H°)",     0, 179, h["h_min"], 1,
                              help="Lower bound of accepted hue — shift if dots aren't detected")
        new_h_max = st.slider("Hue Max (H°)",     0, 179, h["h_max"], 1,
                              help="Upper bound of accepted hue")
        new_s_min = st.slider("Saturation Min",   0, 255, h["s_min"], 1,
                              help="Raise to reject washed-out/white areas")
        new_s_max = st.slider("Saturation Max",   0, 255, h["s_max"], 1)
        new_v_min = st.slider("Value (Brightness) Min", 0, 255, h["v_min"], 1,
                              help="Raise to reject dark shadows")
        new_v_max = st.slider("Value Max",        0, 255, h["v_max"], 1)

        st.markdown('<div class="section-label" style="margin-top:12px;">Blob Area (px²)</div>',
                    unsafe_allow_html=True)
        a = st.session_state.area
        new_min_a = st.slider("Min blob area",   10, 3000, a["min_area"], 10,
                              help="Increase to reject small noise specks")
        new_max_a = st.slider("Max blob area",  200, 25000, a["max_area"], 100,
                              help="Decrease to reject large false-positive regions")

        if st.button("✅ Apply Manual HSV", use_container_width=True):
            # Write slider values back into session state
            st.session_state.hsv = dict(
                h_min=new_h_min, h_max=new_h_max,
                s_min=new_s_min, s_max=new_s_max,
                v_min=new_v_min, v_max=new_v_max,
            )
            st.session_state.area = dict(min_area=new_min_a, max_area=new_max_a)
            st.session_state.hsv_source = "manual"
            st.session_state.centroid_history.clear()   # clear smooth buffer after parameter change
            st.success("Manual HSV applied")
            st.rerun()

        if st.button("↩ Reset to Defaults", use_container_width=True):
            st.session_state.hsv = dict(h_min=90, h_max=130, s_min=80, s_max=255, v_min=50, v_max=255)
            st.session_state.area = dict(min_area=200, max_area=8000)
            st.session_state.hsv_source = "default"
            st.session_state.centroid_history.clear()
            st.rerun()

    with col_preview:
        st.markdown('<div class="section-label">Live Mask Preview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72em;color:#2a5a7a;margin-bottom:10px;">
            Capture a frame below — the mask shows what the AI sees (white = detected dot region).
            You want clean white blobs, minimal noise.
        </div>
        """, unsafe_allow_html=True)

        cal_cam = st.camera_input("Calibration Camera", label_visibility="collapsed",
                                  key="cal_camera")
        if cal_cam is not None:
            bgr = camera_to_bgr(cal_cam)
            if bgr is not None:
                try:
                    # Apply current (possibly slider-edited) HSV to a temporary mask
                    temp_hsv = dict(
                        h_min=new_h_min, h_max=new_h_max,
                        s_min=new_s_min, s_max=new_s_max,
                        v_min=new_v_min, v_max=new_v_max,
                    )
                    mask = _build_mask_internal(
                        bgr,
                        temp_hsv["h_min"], temp_hsv["h_max"],
                        temp_hsv["s_min"], temp_hsv["s_max"],
                        temp_hsv["v_min"], temp_hsv["v_max"],
                    )
                    # Convert binary mask (0/255) to RGB for display
                    # White = detected, black = rejected
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

                    # Side-by-side: original + mask
                    combined = np.hstack([
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),   # original in RGB
                        mask_rgb,                                 # mask preview
                    ])
                    st.image(combined, caption="Original (left) · Mask (right)", use_container_width=True)

                    # Count blobs in current mask for quick feedback
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = sum(1 for c in cnts
                                if new_min_a <= cv2.contourArea(c) <= new_max_a
                                and cv2.arcLength(c, True) > 0
                                and 4 * math.pi * cv2.contourArea(c) / (cv2.arcLength(c, True)**2) >= 0.20)
                    st.markdown(
                        f"<div style='font-size:0.8em;color:#5bc4f5;margin-top:8px;'>"
                        f"✓ {valid} valid blob(s) detected with current settings"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Mask preview error: {e}")

        st.markdown("""
        <div style="font-size:0.72em;color:#1e4a6a;margin-top:16px;line-height:1.9;">
            <b style="color:#3a7aaa;">Hue reference guide:</b><br>
            Red: 0–10 or 160–179 &nbsp;·&nbsp; Orange: 10–25<br>
            Yellow: 25–35 &nbsp;·&nbsp; Green: 35–85 &nbsp;·&nbsp; Cyan: 85–100<br>
            Blue: 100–130 &nbsp;·&nbsp; Purple: 130–160
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────────────
elif page == "About":

    st.markdown("""
    <div class="page-header">About Suterup</div>
    <div class="page-sub">TECH STACK · SCORING GUIDE · TIPS · VERSION HISTORY</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Scoring Guide", "Tech Stack", "Version Notes"])

    with tab1:
        st.markdown("""
        <div class="status-card">
            <div style="font-size:0.85em;font-weight:600;color:#5bc4f5;margin-bottom:12px;letter-spacing:0.08em;">
                SUTURE SCORING RUBRIC
            </div>

            <table style="width:100%;border-collapse:collapse;font-size:0.8em;">
                <tr style="color:#1e4a6a;border-bottom:1px solid #0e2030;">
                    <td style="padding:6px 10px;">Range (px)</td>
                    <td style="padding:6px 10px;">Grade</td>
                    <td style="padding:6px 10px;">Score</td>
                    <td style="padding:6px 10px;">Clinical Meaning</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#4a8aaa;">&lt; 40px</td>
                    <td style="padding:6px 10px;" class="grade-danger">TOO CLOSE</td>
                    <td style="padding:6px 10px;color:#4a8aaa;">0–55</td>
                    <td style="padding:6px 10px;color:#3a6a7a;">Risk of tissue necrosis</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#4a8aaa;">40–60px</td>
                    <td style="padding:6px 10px;" class="grade-short">SHORT</td>
                    <td style="padding:6px 10px;color:#4a8aaa;">55–77</td>
                    <td style="padding:6px 10px;color:#3a6a7a;">Slightly under-spaced</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#4a8aaa;">60–120px</td>
                    <td style="padding:6px 10px;" class="grade-optimal">OPTIMAL ✓</td>
                    <td style="padding:6px 10px;color:#4a8aaa;">77–100</td>
                    <td style="padding:6px 10px;color:#3a6a7a;">Ideal suture spacing</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#4a8aaa;">120–160px</td>
                    <td style="padding:6px 10px;" class="grade-wide">WIDE</td>
                    <td style="padding:6px 10px;color:#4a8aaa;">55–77</td>
                    <td style="padding:6px 10px;color:#3a6a7a;">Wound gap risk</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#4a8aaa;">&gt; 160px</td>
                    <td style="padding:6px 10px;" class="grade-danger">TOO WIDE</td>
                    <td style="padding:6px 10px;color:#4a8aaa;">0–55</td>
                    <td style="padding:6px 10px;color:#3a6a7a;">Closure failure risk</td>
                </tr>
            </table>

            <div style="margin-top:14px;font-size:0.74em;color:#2a5a7a;">
                * Pixel distances are calibrated for a standard webcam at ~25–35cm from the suture model.
                Distances scale with camera distance — move closer for finer measurements.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="status-card">
            <div style="font-size:0.85em;font-weight:600;color:#5bc4f5;margin-bottom:12px;letter-spacing:0.08em;">
                TECHNOLOGY STACK
            </div>
            <div style="font-size:0.78em;color:#5a8aaa;line-height:2.2;">
                <b style="color:#7aaac8;">Streamlit 1.x</b> — Web framework · browser UI, state management, camera widget<br>
                <b style="color:#7aaac8;">OpenCV (cv2)</b> — Image processing · HSV masking, contour detection, annotations<br>
                <b style="color:#7aaac8;">NumPy</b> — Pixel math · array operations on frame data<br>
                <b style="color:#7aaac8;">Pillow (PIL)</b> — Image format conversion between OpenCV and Streamlit<br>
                <b style="color:#7aaac8;">Python stdlib</b> — math, json, collections, datetime — no paid dependencies<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="status-card">
            <b style="color:#5bc4f5;">V5.0</b> — Streamlit migration, multi-page nav, mask preview, dark UI<br>
            <b style="color:#3a6a8a;">V4.0</b> — Auto-HSV 6-pass, temporal smoothing, Gradio live stream<br>
            <b style="color:#3a6a8a;">V3.x</b> — Manual HSV, basic blob detection, static image upload<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;font-size:0.65em;color:#0f2535;padding:10px 0;">
        SUTERUP V5.0 · Surgical Suture Training Platform · Asrith Maruvada · Beta
    </div>
    """, unsafe_allow_html=True)
