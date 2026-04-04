"""
╔══════════════════════════════════════════════════╗
║         SUTERUP V0.2 — Surgical Training          ║
║         Streamlit Cloud Edition                   ║
╚══════════════════════════════════════════════════╝

#use this to run python -m streamlit run "c:/Users/16477/Desktop/All My Projects/sum/test.py"
HOW TO DEPLOY:
  1. Push this file to a GitHub repo
  2. Go to share.streamlit.io
  3. Connect your repo and deploy!


HOW TO USE:
  - Allow camera access in your browser
  - Point camera at blue suture dots
  - Adjust HSV sliders if dots aren't detected
  - Press Start Run, then Log Stitches, then End Run
"""


import streamlit as st
import cv2
import numpy as np
import math
from datetime import datetime
from PIL import Image


# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suterup V0.2 — Surgical Training",
    page_icon="🩺",
    layout="wide"
)


# ─────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a120a; }
    .stApp { background-color: #0a120a; color: #dce8dc; }
    h1, h2, h3 { color: #00c98d; font-family: monospace; }
    .metric-box {
        background: #0f1f0f;
        border: 1px solid #1e4a1e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-family: monospace;
    }
    .grade-optimal { color: #00c98d; font-weight: bold; font-size: 1.2em; }
    .grade-short   { color: #ffdc00; font-weight: bold; font-size: 1.2em; }
    .grade-wide    { color: #ffdc00; font-weight: bold; font-size: 1.2em; }
    .grade-bad     { color: #e05252; font-weight: bold; font-size: 1.2em; }
    .stitch-log {
        background: #0f1f0f;
        border: 1px solid #1e4a1e;
        border-radius: 8px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.85em;
        max-height: 300px;
        overflow-y: auto;
    }
    .header-bar {
        background: #0f1f0f;
        border: 1px solid #00c98d;
        border-radius: 8px;
        padding: 10px 20px;
        font-family: monospace;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────
if "run_active" not in st.session_state:
    st.session_state.run_active = False
if "run_stitches" not in st.session_state:
    st.session_state.run_stitches = []
if "session_total" not in st.session_state:
    st.session_state.session_total = 0
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "last_centroids" not in st.session_state:
    st.session_state.last_centroids = []


# ─────────────────────────────────────────────────────
#  SPATIAL MATH
# ─────────────────────────────────────────────────────


def seg_dist(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def precision(d):
    return max(0, int(100 - abs(d - 90) / 0.9))


def grade(d):
    if d < 40:   return "TOO CLOSE", "#e05252", "Too close — increase stitch width."
    if d < 60:   return "SHORT",     "#ffdc00", "Slightly narrow — aim wider."
    if d <= 120: return "OPTIMAL",   "#00c98d", "Excellent! Within ideal range."
    if d <= 160: return "WIDE",      "#ffdc00", "Slightly wide — bring closer."
    return            "TOO WIDE",   "#e05252", "Too wide — reduce stitch distance."


# ─────────────────────────────────────────────────────
#  MASK + DETECTION
# ─────────────────────────────────────────────────────


def build_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def detect_blobs(frame, h_min, h_max, s_min, s_max, v_min, v_max, min_area, max_area):
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
        return pts, mask
    pts = sorted(pts, key=lambda p: p[0])
    ordered = [pts.pop(0)]
    while pts:
        last = ordered[-1]
        nearest_idx = min(range(len(pts)),
                          key=lambda i: math.sqrt((pts[i][0]-last[0])**2 + (pts[i][1]-last[1])**2))
        ordered.append(pts.pop(nearest_idx))
    return ordered[:10], mask


# ─────────────────────────────────────────────────────
#  DRAWING ON FRAME
# ─────────────────────────────────────────────────────


PT_COLORS_BGR = [
    (141, 201, 0), (255, 140, 66), (255, 200, 77),
    (255, 120, 200), (255, 219, 77), (130, 255, 140),
    (60, 180, 255), (255, 80, 60), (200, 80, 220), (180, 255, 80),
]


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


def annotate_frame(frame, centroids):
    display = frame.copy()
    for i in range(len(centroids) - 1):
        p1, p2 = centroids[i], centroids[i+1]
        d = seg_dist(p1, p2)
        g_label, g_hex, _ = grade(d)
        # Convert hex to BGR
        h = g_hex.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        color = (b, g, r)
        draw_dashed(display, p1, p2, color, 2)
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        sc = precision(d)
        text = f"{d:.0f}px {sc}/100"
        cv2.putText(display, text, (mid[0]-20, mid[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for i, pt in enumerate(centroids):
        color = PT_COLORS_BGR[i % len(PT_COLORS_BGR)]
        cv2.circle(display, pt, 14, color, 1)
        cv2.circle(display, pt, 8,  color, -1)
        cv2.circle(display, pt, 3,  (220,240,232), -1)
        cv2.putText(display, f"P{i+1}", (pt[0]+16, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    return display


# ─────────────────────────────────────────────────────
#  UI LAYOUT
# ─────────────────────────────────────────────────────


st.markdown("""
<div class="header-bar">
    <span style="color:#00c98d; font-size:1.3em;">⚕ SUTERUP V0.2</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span style="color:#dce8dc;">Surgical Training Platform</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span style="color:#7aaa7a; font-size:0.85em;">by Asrith Maruvada</span>
</div>
""", unsafe_allow_html=True)


col_cam, col_controls = st.columns([3, 1])


with col_controls:
    st.markdown("### ⚙️ HSV Tuner")
    st.caption("Adjust until your dots are detected. Blue dots work best out of the box.")
    h_min = st.slider("H min", 0, 179, 90)
    h_max = st.slider("H max", 0, 179, 130)
    s_min = st.slider("S min", 0, 255, 80)
    s_max = st.slider("S max", 0, 255, 255)
    v_min = st.slider("V min", 0, 255, 50)
    v_max = st.slider("V max", 0, 255, 255)
    min_area = st.slider("Min blob area (px²)", 50, 2000, 200)
    max_area = st.slider("Max blob area (px²)", 1000, 20000, 8000)


    st.markdown("---")
    st.markdown("### 🎮 Controls")


    run_btn = st.button("▶ Start Run",  disabled=st.session_state.run_active, use_container_width=True)
    log_btn = st.button("📝 Log Stitches", disabled=not st.session_state.run_active, use_container_width=True)
    end_btn = st.button("⏹ End Run",    disabled=not st.session_state.run_active, use_container_width=True)
    rst_btn = st.button("🔄 Reset Session", use_container_width=True)


    if run_btn:
        st.session_state.run_active   = True
        st.session_state.run_stitches = []
        st.session_state.log_messages.append("🟢 Run started — position dots and click Log Stitches")
        st.rerun()


    if rst_btn:
        st.session_state.run_active    = False
        st.session_state.run_stitches  = []
        st.session_state.session_total = 0
        st.session_state.log_messages  = ["🔄 Session reset."]
        st.session_state.last_centroids = []
        st.rerun()


with col_cam:
    st.markdown("### 📷 Camera Feed")
    st.caption("Allow camera access when prompted. Point at blue suture dots.")


    img_file = st.camera_input("", label_visibility="collapsed")


    if img_file is not None:
        # Convert to OpenCV format
        pil_img = Image.open(img_file)
        frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


        # Detect
        centroids, mask = detect_blobs(
            frame, h_min, h_max, s_min, s_max, v_min, v_max, min_area, max_area
        )
        st.session_state.last_centroids = centroids


        # Annotate
        annotated = annotate_frame(frame, centroids)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


        tab1, tab2 = st.tabs(["📹 Annotated View", "🎭 Mask View"])
        with tab1:
            st.image(annotated_rgb, use_container_width=True)
        with tab2:
            st.image(mask, use_container_width=True, caption="Blue dots should appear as white blobs")


        # Stats row
        if len(centroids) >= 2:
            dists  = [seg_dist(centroids[i], centroids[i+1]) for i in range(len(centroids)-1)]
            scores = [precision(d) for d in dists]
            avg_d  = sum(dists) / len(dists)
            avg_sc = int(sum(scores) / len(scores))
            last_d = dists[-1]
            g_label, g_hex, feedback = grade(last_d)


            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-box"><div style="color:#7aaa7a;font-size:0.8em;">POINTS</div><div style="color:#00c98d;font-size:1.5em;">{len(centroids)}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div style="color:#7aaa7a;font-size:0.8em;">AVG SCORE</div><div style="color:#00c98d;font-size:1.5em;">{avg_sc}/100</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div style="color:#7aaa7a;font-size:0.8em;">AVG DIST</div><div style="color:#00c98d;font-size:1.5em;">{avg_d:.0f}px</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-box"><div style="color:#7aaa7a;font-size:0.8em;">LAST GRADE</div><div style="color:{g_hex};font-size:1.2em;">{g_label}</div></div>', unsafe_allow_html=True)


            st.info(f"💬 {feedback}")


            # Log stitches action
            if log_btn:
                logged = 0
                for i in range(len(centroids) - 1):
                    d  = seg_dist(centroids[i], centroids[i+1])
                    sc = precision(d)
                    g_label_l, _, _ = grade(d)
                    st.session_state.run_stitches.append({"d": d, "sc": sc})
                    st.session_state.session_total += 1
                    logged += 1
                    tag = " [START]" if len(st.session_state.run_stitches) == 1 else ""
                    st.session_state.log_messages.append(
                        f"📌{tag} #{len(st.session_state.run_stitches)} | {d:.1f}px | {g_label_l} | {sc}/100"
                    )
                st.session_state.log_messages.append(f"✅ {logged} stitch(es) logged")
                st.rerun()


            # End run action
            if end_btn:
                for i in range(len(centroids) - 1):
                    d  = seg_dist(centroids[i], centroids[i+1])
                    sc = precision(d)
                    g_label_e, _, _ = grade(d)
                    st.session_state.run_stitches.append({"d": d, "sc": sc})
                    st.session_state.session_total += 1
                run_s = st.session_state.run_stitches
                if run_s:
                    s_scores = [s["sc"] for s in run_s]
                    s_dists  = [s["d"]  for s in run_s]
                    optimal  = sum(1 for s in run_s if 60 <= s["d"] <= 120)
                    st.session_state.log_messages.append("━━━━━━━━━━━━━━━━━━━━━━━━")
                    st.session_state.log_messages.append("🏁 RUN COMPLETE — SUMMARY")
                    st.session_state.log_messages.append(f"Stitches: {len(run_s)}")
                    st.session_state.log_messages.append(f"Avg score: {int(sum(s_scores)/len(s_scores))}/100")
                    st.session_state.log_messages.append(f"Best: {max(s_scores)}/100  Worst: {min(s_scores)}/100")
                    st.session_state.log_messages.append(f"Optimal: {optimal}/{len(run_s)}")
                    st.session_state.log_messages.append(f"Avg dist: {sum(s_dists)/len(s_dists):.1f}px")
                    st.session_state.log_messages.append("━━━━━━━━━━━━━━━━━━━━━━━━")
                st.session_state.run_active   = False
                st.session_state.run_stitches = []
                st.rerun()


        else:
            if len(centroids) == 0:
                st.warning("⚠️ No blue dots detected. Try adjusting HSV sliders or improve lighting.")
            else:
                st.warning("⚠️ Only 1 dot detected. Need at least 2 to measure stitch spacing.")


        if log_btn and len(centroids) < 2:
            st.error("Need at least 2 detected points to log stitches.")
        if end_btn and len(centroids) < 2:
            st.session_state.run_active = False
            st.rerun()


    else:
        st.info("👆 Click the camera button above to capture a frame for analysis.")
        st.markdown("""
        **Setup tips:**
        - Place small **blue stickers** or **blue nail polish dots** at each suture point
        - Ensure good lighting — avoid shadows on the dots
        - Keep dots between 1–3cm apart for realistic suture spacing
        - Press **M** in the mask view to tune detection if dots aren't found
        """)


# ─────────────────────────────────────────────────────
#  STITCH LOG + SESSION STATS
# ─────────────────────────────────────────────────────
st.markdown("---")
col_log, col_stats = st.columns([2, 1])


with col_log:
    st.markdown("### 📋 Stitch Log")
    if st.session_state.log_messages:
        log_html = "<br>".join(st.session_state.log_messages[-30:])
        st.markdown(f'<div class="stitch-log">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.caption("Start a run and log stitches — they'll appear here.")


with col_stats:
    st.markdown("### 📊 Session Stats")
    st.markdown(f'<div class="metric-box"><div style="color:#7aaa7a;">TOTAL STITCHES</div><div style="color:#00c98d;font-size:2em;">{st.session_state.session_total}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    status = "🔴 RECORDING" if st.session_state.run_active else "⚪ STANDBY"
    st.markdown(f'<div class="metric-box"><div style="color:#7aaa7a;">STATUS</div><div style="font-size:1.2em;">{status}</div></div>', unsafe_allow_html=True)


st.markdown("---")
st.markdown('<div style="text-align:center;color:#3a6a3a;font-family:monospace;font-size:0.8em;">SUTERUP V0.2 · Surgical Training Platform · Asrith Maruvada</div>', unsafe_allow_html=True)



