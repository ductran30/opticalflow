# app.py — Video Stabilization (Lucas–Kanade & FlowNet2)
# ======================================================
# - Ghi video bằng imageio-ffmpeg (libx264, yuv420p) -> ổn định trên macOS/Streamlit
# - Nút tạo video so sánh (Gốc | Đã ổn định)
# - PSNR/SSIM theo từng frame + Shakiness trước/sau
# - Dùng st.session_state để giữ kết quả sau rerun
# - Có CLI fallback
# ======================================================

import os
import sys
import math
import argparse
import tempfile
from typing import List, Tuple

import cv2
import numpy as np

# tqdm (CLI progress) — optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# Streamlit — optional
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    st = None
    STREAMLIT_AVAILABLE = False

# imageio + ffmpeg — bắt buộc cho ghi video
try:
    import imageio.v3 as iio
    import imageio_ffmpeg
    IMAGEIO_AVAILABLE = True
except Exception:
    iio = None
    IMAGEIO_AVAILABLE = False


# -------------------- Utils --------------------
def ensure_even_size(w: int, h: int) -> Tuple[int, int]:
    if w % 2: w -= 1
    if h % 2: h -= 1
    return max(2, w), max(2, h)

def compute_safe_crop_fixed(W: int, H: int, drop_ratio: float = 0.05, min_side: int = 32) -> Tuple[int, int, int, int]:
    """Fixed crop tỉ lệ theo mỗi cạnh, trả (x, y, w, h), luôn even size."""
    left  = min(int(round(W * drop_ratio)), (W - min_side)//2)
    top   = min(int(round(H * drop_ratio)), (H - min_side)//2)
    cx, cy = left, top
    cw = max(min_side, W - 2*left)
    ch = max(min_side, H - 2*top)
    cw, ch = ensure_even_size(cw, ch)
    return cx, cy, cw, ch

def read_frames_resize(path: str, resize_short: int) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]
        s = resize_short / float(min(h, w))
        f = cv2.resize(f, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError("Empty video or unsupported format.")
    return frames, fps

def write_video_imageio(frames: List[np.ndarray], fps: float, out_path: str):
    """Ghi MP4 bằng imageio-ffmpeg (libx264, yuv420p) — ổn định trên macOS/Streamlit."""
    if not IMAGEIO_AVAILABLE:
        raise RuntimeError("Cần imageio + imageio-ffmpeg. Cài: pip install imageio imageio-ffmpeg")
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    H, W = rgb_frames[0].shape[:2]
    W2, H2 = ensure_even_size(W, H)
    if (W2, H2) != (W, H):
        rgb_frames = [cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA) for f in rgb_frames]
    iio.imwrite(out_path, rgb_frames, fps=fps, codec="libx264", pixelformat="yuv420p")


# -------------------- Lucas–Kanade --------------------
def estimate_affine_partial(prev_gray: np.ndarray, cur_gray: np.ndarray, max_pts: int = 200) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_pts, qualityLevel=0.01, minDistance=30, blockSize=3)
    if pts is None or len(pts) < 10:
        return np.array([[1,0,0],[0,1,0]], np.float32)
    nxt, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
    ok = (stt.flatten() == 1)
    if ok.sum() < 10:
        return np.array([[1,0,0],[0,1,0]], np.float32)
    src = pts[ok].reshape(-1,2)
    dst = nxt[ok].reshape(-1,2)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return M if M is not None else np.array([[1,0,0],[0,1,0]], np.float32)

def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    w = 2*radius + 1
    f = np.ones(w) / w
    pad = np.pad(curve, (radius, radius), 'edge')
    return np.convolve(pad, f, mode='same')[radius:-radius]

def smooth_traj(transforms: np.ndarray, radius: int) -> np.ndarray:
    traj = np.cumsum(transforms, axis=0)
    sm = traj.copy()
    for i in range(3):
        sm[:, i] = moving_average(traj[:, i], radius)
    diff = sm - traj
    return transforms + diff

def stabilize_lk(frames: List[np.ndarray], fps: float, smooth_k: int, drop_ratio: float, progress=None) -> Tuple[str, bytes]:
    H, W = frames[0].shape[:2]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    transforms = []
    for i in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        M = estimate_affine_partial(prev_gray, cur_gray)
        dx, dy = float(M[0,2]), float(M[1,2])
        da = float(math.atan2(M[1,0], M[0,0]))
        transforms.append([dx, dy, da])
        prev_gray = cur_gray
        if progress is not None:
            try: progress(i / (len(frames)-1))
            except Exception: pass

    if not transforms:
        raise RuntimeError("Not enough frames to stabilize.")
    T = np.array(transforms, dtype=np.float32)
    T_s = smooth_traj(T, radius=max(1, smooth_k))

    cx, cy, cw, ch = compute_safe_crop_fixed(W, H, drop_ratio=drop_ratio)

    out_frames = [frames[0][cy:cy+ch, cx:cx+cw]]
    for i, (dx, dy, da) in enumerate(T_s):
        M = np.array([[math.cos(da), -math.sin(da), dx],
                      [math.sin(da),  math.cos(da), dy]], dtype=np.float32)
        wf = cv2.warpAffine(frames[i+1], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out_frames.append(wf[cy:cy+ch, cx:cx+cw])

    tmp_out = tempfile.mktemp(suffix="_lk.mp4")
    write_video_imageio(out_frames, fps, tmp_out)
    with open(tmp_out, 'rb') as fh:
        data = fh.read()
    return tmp_out, data


# -------------------- FlowNet2 (PTLFlow) --------------------
def _load_flownet2_model():
    try:
        import torch, ptlflow
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("FlowNet2 cần ptlflow + torch. Cài: pip install ptlflow torch torchvision") from e
    device = 'cuda'
    try:
        import torch
        if not torch.cuda.is_available():
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    except Exception:
        device = 'cpu'
    import ptlflow
    model = ptlflow.get_model('flownet2', ckpt_path='things').to(device).eval()
    return model, device

def estimate_flow_flownet2(prev_bgr: np.ndarray, cur_bgr: np.ndarray, model, device: str) -> np.ndarray:
    import torch
    from ptlflow.utils.io_adapter import IOAdapter
    prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    cur_rgb  = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2RGB)
    io = IOAdapter(model, prev_rgb.shape[:2])
    inputs = io.prepare_inputs([prev_rgb, cur_rgb])
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(inputs)
        flow = pred['flows'][0, 0]  # (2,H,W)
        flow = flow.permute(1, 2, 0).detach().cpu().numpy()
    return flow

def flow_to_affine(flow: np.ndarray, stride: int = 16, rth: float = 3.0, max_iter: int = 2000) -> np.ndarray:
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    src = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    dst = src + flow[ys, xs].reshape(-1, 2).astype(np.float32)
    M, _ = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=rth, maxIters=max_iter, refineIters=50)
    return M if M is not None else np.array([[1,0,0],[0,1,0]], np.float32)

def stabilize_flownet2(frames: List[np.ndarray], fps: float, smooth_k: int, drop_ratio: float, progress=None) -> Tuple[str, bytes]:
    model, device = _load_flownet2_model()
    H, W = frames[0].shape[:2]
    aff = [np.array([[1,0,0],[0,1,0]], np.float32)]
    prev = frames[0]
    for i in range(1, len(frames)):
        cur = frames[i]
        flow = estimate_flow_flownet2(prev, cur, model, device)
        M = flow_to_affine(flow, stride=16, rth=3.0)
        aff.append(M); prev = cur
        if progress is not None:
            try: progress(i / (len(frames)-1))
            except Exception: pass

    # Làm mượt vector 6D (trung bình cửa sổ)
    vec = np.array([[M[0,0], M[0,1], M[1,0], M[1,1], M[0,2], M[1,2]] for M in aff], dtype=np.float32)
    smoothed = []
    for i in range(len(vec)):
        s = vec[max(0, i - smooth_k):min(len(vec), i + smooth_k + 1)].mean(axis=0)
        smoothed.append(np.array([[s[0], s[1], s[4]], [s[2], s[3], s[5]]], dtype=np.float32))

    cx, cy, cw, ch = compute_safe_crop_fixed(W, H, drop_ratio=drop_ratio)

    out_frames = []
    for f, D in zip(frames, smoothed):
        wf = cv2.warpAffine(f, D, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out_frames.append(wf[cy:cy+ch, cx:cx+cw])

    tmp_out = tempfile.mktemp(suffix="_flownet2.mp4")
    write_video_imageio(out_frames, fps, tmp_out)
    with open(tmp_out, 'rb') as fh:
        data = fh.read()
    return tmp_out, data


# -------------------- Metrics & Side-by-Side --------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    ax = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    bx = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    gk, sig = (11, 11), 1.5
    mu1 = cv2.GaussianBlur(ax, gk, sig)
    mu2 = cv2.GaussianBlur(bx, gk, sig)
    s1 = cv2.GaussianBlur(ax * ax, gk, sig) - mu1 * mu1
    s2 = cv2.GaussianBlur(bx * bx, gk, sig) - mu2 * mu2
    s12 = cv2.GaussianBlur(ax * bx, gk, sig) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2)
    return float((num / den).mean())

def shakiness(video_path: str, max_pts: int = 500) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return float('nan')
    ok, prev = cap.read()
    if not ok: cap.release(); return float('nan')
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags = []
    while True:
        ok, cur = cap.read()
        if not ok: break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_pts, qualityLevel=0.01, minDistance=8, blockSize=7)
        if pts is None or len(pts) < 10:
            prev_gray = cur_gray; continue
        nxt, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, winSize=(21,21), maxLevel=3)
        okm = (stt.flatten() == 1)
        if okm.sum() < 10:
            prev_gray = cur_gray; continue
        src = pts[okm].reshape(-1,2)
        dst = nxt[okm].reshape(-1,2)
        M, _ = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=3.0, maxIters=2000)
        if M is None:
            prev_gray = cur_gray; continue
        tx, ty = float(M[0,2]), float(M[1,2])
        mags.append((tx*tx + ty*ty) ** 0.5)
        prev_gray = cur_gray
    cap.release()
    return float(np.std(mags)) if mags else float('nan')

def make_side_by_side(original_path: str, stabilized_path: str, out_compare_path: str):
    """Ghép ngang: Gốc | Đã ổn định (có vạch trắng phân cách)."""
    cap0 = cv2.VideoCapture(original_path)
    cap1 = cv2.VideoCapture(stabilized_path)
    if not (cap0.isOpened() and cap1.isOpened()):
        raise RuntimeError("Không mở được 1 trong 2 video để so sánh.")
    n0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n  = min(n0, n1)
    frames = []
    for _ in range(n):
        r0, f0 = cap0.read(); r1, f1 = cap1.read()
        if not (r0 and r1): break
        h1, w1 = f1.shape[:2]
        f0r = cv2.resize(f0, (w1, h1), interpolation=cv2.INTER_AREA)
        sep = np.full((h1, 4, 3), 255, dtype=np.uint8)  # separator
        combo = np.hstack([f0r, sep, f1])
        frames.append(combo)
    cap0.release(); cap1.release()
    if not frames:
        raise RuntimeError("Không tạo được frame so sánh.")
    fps = cv2.VideoCapture(stabilized_path).get(cv2.CAP_PROP_FPS) or 30.0
    write_video_imageio(frames, fps, out_compare_path)


# -------------------- Streamlit UI --------------------
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Video Stabilization — LK & FlowNet2", layout="wide")
    st.title("🎥 Video Stabilization — Lucas–Kanade & FlowNet2")
    st.caption("Upload → chọn phương pháp → ổn định → so sánh & đánh giá. Trên macOS (không CUDA), LK chạy nhanh; FlowNet2 có thể chậm (MPS/CPU).")

    # Khởi tạo session_state để lưu kết quả sau rerun
    for k, v in {
        "in_path": None,
        "out_path": None,
        "compare_path": None,
        "psnrs": None,
        "ssims": None,
        "shak_o": None,
        "shak_s": None,
        "done": False,
    }.items():
        st.session_state.setdefault(k, v)

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        method = st.radio("Phương pháp", ["Lucas–Kanade (Feature)", "FlowNet2 (PTLFlow)"])
        resize_short = st.select_slider("Resize ngắn cạnh", options=[360, 480, 720, 1080], value=720)
        smooth_k = st.slider("Cửa sổ làm mượt (k)", 10, 80, 30, step=2)
        drop_ratio = st.slider("Crop mỗi cạnh (%)", 0, 15, 5, step=1) / 100.0

    uploaded = st.file_uploader("Chọn video đầu vào (MP4 khuyến nghị)", type=["mp4","mov","avi","mkv"], accept_multiple_files=False)
    if uploaded: st.video(uploaded)

    col1, col2 = st.columns([1,1])
    run = st.button("🚀 Stabilize", type="primary", use_container_width=True, disabled=not uploaded)

    # === XỬ LÝ ỔN ĐỊNH (BÊN TRONG KHỐI RUN) ===
    if run and uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            in_path = tmp.name

        try:
            frames, fps = read_frames_resize(in_path, resize_short)
        except Exception as e:
            st.error(f"Không đọc được video: {e}")
            st.stop()

        st.info("Đang xử lý… (FlowNet2 lần đầu có thể tải ~600MB weights)")

        def _cb(p):
            st.session_state.setdefault('_p', st.progress(0.0))
            st.session_state['_p'].progress(min(1.0, max(0.0, float(p))))

        try:
            if method.startswith("Lucas"):
                out_path, _ = stabilize_lk(frames, fps, smooth_k, drop_ratio, progress=_cb)
            else:
                out_path, _ = stabilize_flownet2(frames, fps, smooth_k, drop_ratio, progress=_cb)
        except Exception as e:
            st.error(f"Lỗi xử lý: {e}")
            st.stop()

        st.success("Hoàn tất!")

        # LƯU VÀO SESSION_STATE để không mất sau rerun
        st.session_state.in_path = in_path
        st.session_state.out_path = out_path
        st.session_state.compare_path = None
        st.session_state.psnrs = None
        st.session_state.ssims = None
        st.session_state.shak_o = None
        st.session_state.shak_s = None
        st.session_state.done = True

        with col1:
            st.subheader("Gốc")
            st.video(in_path)
        with col2:
            st.subheader("Đã ổn định")
            st.video(out_path)

        st.download_button("⬇️ Tải video ổn định",
                           data=open(out_path, "rb").read(),
                           file_name=os.path.basename(out_path),
                           mime="video/mp4",
                           use_container_width=True)

    # === KHU VỰC KẾT QUẢ (LUÔN HIỂN THỊ KHI ĐÃ XỬ LÝ) ===
    if st.session_state.done and st.session_state.in_path and st.session_state.out_path:
        st.write("---")
        st.subheader("🎬 So sánh Side‑by‑Side")

        c1, c2 = st.columns([1,1])
        with c1:
            make_btn = st.button("Tạo video so sánh (Gốc | Đã ổn định)", use_container_width=True)

        if make_btn:
            compare_path = tempfile.mktemp(suffix="_compare.mp4")
            try:
                make_side_by_side(st.session_state.in_path, st.session_state.out_path, compare_path)
                st.session_state.compare_path = compare_path
                st.success("Đã tạo video so sánh.")
            except Exception as e:
                st.error(f"Không thể tạo video so sánh: {e}")

        if st.session_state.compare_path and os.path.exists(st.session_state.compare_path):
            st.video(st.session_state.compare_path)
            st.download_button("⬇️ Tải video so sánh",
                               data=open(st.session_state.compare_path, "rb").read(),
                               file_name=os.path.basename(st.session_state.compare_path),
                               mime="video/mp4",
                               use_container_width=True)

        # -------- Metrics --------
        st.write("---")
        st.subheader("📊 PSNR / SSIM / Shakiness")

        calc = st.button("Tính/hiển thị chỉ số", use_container_width=True)

        if calc:
            cap0 = cv2.VideoCapture(st.session_state.in_path)
            cap1 = cv2.VideoCapture(st.session_state.out_path)
            n = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT), cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
            psnrs, ssims = [], []
            step = max(1, n // 300)  # giới hạn số điểm hiển thị
            i = 0
            while True:
                cap0.set(cv2.CAP_PROP_POS_FRAMES, i)
                cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
                r0, f0 = cap0.read(); r1, f1 = cap1.read()
                if not (r0 and r1): break
                h1, w1 = f1.shape[:2]
                f0r = cv2.resize(f0, (w1, h1), interpolation=cv2.INTER_AREA)
                psnrs.append(psnr(f0r, f1))
                ssims.append(ssim(f0r, f1))
                i += step
                if i >= n: break
            cap0.release(); cap1.release()

            shak_o = shakiness(st.session_state.in_path)
            shak_s = shakiness(st.session_state.out_path)

            st.session_state.psnrs = psnrs
            st.session_state.ssims = ssims
            st.session_state.shak_o = shak_o
            st.session_state.shak_s = shak_s

        # Hiển thị nếu đã tính xong
        if st.session_state.psnrs is not None and st.session_state.ssims is not None:
            m_psnr = float(np.mean(st.session_state.psnrs)) if st.session_state.psnrs else float('nan')
            m_ssim = float(np.mean(st.session_state.ssims)) if st.session_state.ssims else float('nan')
            delta = None
            if st.session_state.shak_o is not None and st.session_state.shak_s is not None \
               and not math.isnan(st.session_state.shak_o) and not math.isnan(st.session_state.shak_s):
                delta = st.session_state.shak_o - st.session_state.shak_s

            m1, m2, m3 = st.columns(3)
            m1.metric("PSNR (mean)", f"{m_psnr:.2f} dB")
            m2.metric("SSIM (mean)", f"{m_ssim:.4f}")
            if delta is not None:
                m3.metric("Δ Shakiness (↓ tốt)", f"{delta:+.3f}",
                          help=f"orig={st.session_state.shak_o:.3f} | stab={st.session_state.shak_s:.3f}")

            if st.session_state.psnrs:
                st.line_chart({"PSNR (dB)": st.session_state.psnrs}, height=220)
            if st.session_state.ssims:
                st.line_chart({"SSIM": st.session_state.ssims}, height=220)

        st.write("")
        if st.button("🔄 Làm lại từ đầu", type="secondary"):
            for k in ["in_path","out_path","compare_path","psnrs","ssims","shak_o","shak_s","done"]:
                st.session_state[k] = (False if k=="done" else None)
            st.experimental_rerun()


# -------------------- CLI fallback --------------------
def run_cli():
    parser = argparse.ArgumentParser(description="Video Stabilization (LK & FlowNet2) — CLI")
    parser.add_argument('--input', required=False, help='Input video path')
    parser.add_argument('--output', required=False, help='Output stabilized video (mp4)')
    parser.add_argument('--method', default='lk', choices=['lk','flownet2'], help='Method')
    parser.add_argument('--resize-short', type=int, default=720, help='Short-side resize')
    parser.add_argument('--smooth-k', type=int, default=30, help='Smoothing window')
    parser.add_argument('--drop-ratio', type=float, default=0.05, help='Crop per side ratio [0..0.2]')
    parser.add_argument('--metrics', action='store_true', help='Compute PSNR/SSIM/Shakiness')
    parser.add_argument('--compare', action='store_true', help='Export side-by-side comparison video')
    parser.add_argument('--compare-out', type=str, default=None, help='Path for side-by-side video')
    args = parser.parse_args()

    if not args.input or not args.output:
        parser.error("--input và --output là bắt buộc ở CLI.")

    frames, fps = read_frames_resize(args.input, args.resize_short)

    def _p(p):
        pct = int(p * 100)
        if pct % 10 == 0: print(f"Progress: {pct:3d}%", end='\r')

    if args.method == 'lk':
        out_path, _ = stabilize_lk(frames, fps, args.smooth_k, args.drop_ratio, progress=_p)
    else:
        out_path, _ = stabilize_flownet2(frames, fps, args.smooth_k, args.drop_ratio, progress=_p)

    os.replace(out_path, args.output)
    print(f"\n✅ Saved: {args.output}")

    if args.compare:
        cmp_path = args.compare_out or os.path.splitext(args.output)[0] + "_compare.mp4"
        make_side_by_side(args.input, args.output, cmp_path)
        print(f"✅ Side-by-side: {cmp_path}")

    if args.metrics:
        print("Computing metrics…")
        cap0 = cv2.VideoCapture(args.input)
        cap1 = cv2.VideoCapture(args.output)
        n = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT), cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        psnrs, ssims = [], []
        step = max(1, n // 300)
        for i in range(0, n, step):
            cap0.set(cv2.CAP_PROP_POS_FRAMES, i)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
            r0, f0 = cap0.read(); r1, f1 = cap1.read()
            if not (r0 and r1): break
            h1, w1 = f1.shape[:2]
            f0r = cv2.resize(f0, (w1, h1), interpolation=cv2.INTER_AREA)
            psnrs.append(psnr(f0r, f1))
            ssims.append(ssim(f0r, f1))
        cap0.release(); cap1.release()
        shak_o = shakiness(args.input)
        shak_s = shakiness(args.output)
        print(f"Frames compared: {len(psnrs)}")
        if psnrs: print(f"PSNR mean: {np.mean(psnrs):.2f}")
        if ssims: print(f"SSIM mean: {np.mean(ssims):.4f}")
        if (not math.isnan(shak_o)) and (not math.isnan(shak_s)):
            print(f"Shakiness: original={shak_o:.3f} | stabilized={shak_s:.3f} → Δ={shak_o - shak_s:+.3f} (↓ tốt)")


# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    if (not STREAMLIT_AVAILABLE) or any(a.startswith('--') for a in sys.argv[1:]):
        run_cli()
    else:
        print("Streamlit available. Run UI: streamlit run app.py  |  Or CLI: python app.py --help")
