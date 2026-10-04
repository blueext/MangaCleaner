"""
Streamlit app: Webtoon/Manhwa Text Remover
==========================================
Same pipeline as webtoon_text_remover_lama.py (EasyOCR EN/KO/JA + LaMa),
wrapped in a web UI:
  - upload single image, multiple images, or a ZIP archive
  - adjustable OCR / mask / tile settings
  - GPU or CPU mode
  - per-image download + one ZIP for everything

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy: push this file + requirements.txt to a GitHub repo and
choose "New app" on https://share.streamlit.io
"""

import hashlib
import io
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------------ helpers

def slice_image(height, win_h, overlap):
    """Yield (y0, y1) windows covering [0, height) with given overlap."""
    step = win_h - overlap
    y = 0
    while y < height:
        y0 = y
        y1 = min(y + win_h, height)
        yield y0, y1
        if y1 == height:
            break
        y += step


def boxes_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    return ix > 0 and iy > 0 and (ix * iy) > 0.5 * min(
        (ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))


def merge_boxes(boxes, tol=6):
    """Greedy-merge overlapping / near-duplicate boxes."""
    boxes = [tuple(map(int, b)) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            cur = list(a)
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b = boxes[j]
                padded = (cur[0] - tol, cur[1] - tol, cur[2] + tol,
                          cur[3] + tol)
                if boxes_overlap(padded, b):
                    cur = [min(cur[0], b[0]), min(cur[1], b[1]),
                           max(cur[2], b[2]), max(cur[3], b[3])]
                    used[j] = True
                    merged = True
            used[i] = True
            out.append(tuple(cur))
        boxes = out
    return boxes


# ------------------------------------------------------------------ models
# Cached across reruns so Streamlit does not reload EasyOCR/LaMa on every
# widget change.

@st.cache_resource(show_spinner="Loading EasyOCR model …")
def get_reader(languages, gpu):
    import easyocr
    return easyocr.Reader(list(languages), gpu=gpu)


@st.cache_resource(show_spinner="Loading LaMa model …")
def get_lama(device):
    from simple_lama_inpainting import SimpleLama
    return SimpleLama(device=device)


# ------------------------------------------------------------------ pipeline

def detect_text_boxes(image_bgr, reader, win_h, overlap, conf):
    """Run OCR over sliding windows; return merged absolute boxes."""
    h, w = image_bgr.shape[:2]
    all_boxes = []
    windows = list(slice_image(h, win_h, overlap))
    for (y0, y1) in windows:
        strip = image_bgr[y0:y1]
        results = reader.readtext(strip, detail=1, paragraph=False)
        for box, _text, prob in results:
            if prob < conf:
                continue
            pts = np.array(box, dtype=np.float32)
            x0, y_min = pts.min(axis=0)
            x1, y_max = pts.max(axis=0)
            all_boxes.append(
                (int(max(0, x0)), int(y0 + max(0, y_min)),
                 int(min(w, x1)), int(y0 + min(y1 - y0, y_max))))
    return merge_boxes(all_boxes)


def build_mask(shape, boxes, dilate_px):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        pad_x = max(2, dilate_px // 2)
        pad_y = max(2, dilate_px // 3)
        cv2.rectangle(mask,
                      (max(0, x0 - pad_x), max(0, y0 - pad_y)),
                      (min(shape[1] - 1, x1 + pad_x),
                       min(shape[0] - 1, y1 + pad_y)),
                      255, thickness=-1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k)
    return mask


def inpaint_lama_tiled(image_bgr, mask, lama, tile=1024, overlap=128):
    """Inpaint with LaMa in overlapping tiles, cross-fading tile seams."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    x_starts = list(slice_image(w, tile, overlap))
    y_starts = list(slice_image(h, tile, overlap))

    acc = np.zeros((h, w, 3), dtype=np.float64)
    wacc = np.zeros((h, w, 1), dtype=np.float64)

    for (y0, y1) in y_starts:
        for (x0, x1) in x_starts:
            m_tile = mask[y0:y1, x0:x1]
            if not np.any(m_tile):
                continue  # nothing to inpaint here

            img_tile = Image.fromarray(rgb[y0:y1, x0:x1])
            msk_tile = Image.fromarray(m_tile)          # L mode, 0/255
            out_tile = np.asarray(lama(img_tile, msk_tile),
                                  dtype=np.float64)

            wy = np.ones(y1 - y0, dtype=np.float64)
            wx = np.ones(x1 - x0, dtype=np.float64)
            if y0 > 0:
                wy[:overlap] = np.linspace(0, 1, overlap)
            if y1 < h:
                wy[-overlap:] = np.minimum(wy[-overlap:],
                                           np.linspace(1, 0, overlap))
            if x0 > 0:
                wx[:overlap] = np.linspace(0, 1, overlap)
            if x1 < w:
                wx[-overlap:] = np.minimum(wx[-overlap:],
                                           np.linspace(1, 0, overlap))
            weight = (wy[:, None] * wx[None, :])[:, :, None]

            acc[y0:y1, x0:x1] += out_tile * weight
            wacc[y0:y1, x0:x1] += weight

    # skipped tiles keep wacc == 0 -> original pixels preserved
    out = rgb.astype(np.float64)
    np.divide(acc, wacc, out=np.zeros_like(acc), where=wacc > 0)
    result = np.where(wacc > 0, acc, out).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def pil_to_png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ UI

st.set_page_config(page_title="Webtoon Text Remover",
                   page_icon="🖌️", layout="wide")

st.title("🖌️ Webtoon / Manhwa Text Remover")
st.caption("Detects EN/KO/JA text with EasyOCR and erases it with "
           "LaMa inpainting. Upload an image (or many, or a ZIP), "
           "tune the settings, download the clean pages.")

# ---- sidebar settings ----
with st.sidebar:
    st.header("Settings")
    languages = st.multiselect(
        "OCR languages",
        options=["en", "ko", "ja"],
        default=["en", "ko", "ja"])
    gpu = st.toggle("Use GPU (CUDA)", value=False,
                    help="Turn on only if the machine has a CUDA GPU "
                         "(e.g. a Colab-style box). Streamlit Cloud = CPU.")
    conf = st.slider("OCR confidence threshold", 0.0, 1.0, 0.3, 0.05,
                     help="Higher = fewer boxes, may miss faint text.")
    dilate = st.slider("Mask dilation (px)", 0, 40, 9,
                       help="Extra padding around detected text.")
    tile = st.select_slider("LaMa tile size (px)",
                            options=[512, 768, 1024, 1536], value=1024)
    win = st.number_input("OCR window height (px)", 500, 8000, 3000, 500)
    overlap = st.number_input("OCR window overlap (px)", 0, 1500, 300, 50)
    save_mask = st.checkbox("Also export text masks", value=False)
    st.divider()
    st.caption("First run downloads model weights (~350 MB total). "
               "Be patient if you're on free cloud RAM.")

if not languages:
    st.warning("Pick at least one OCR language in the sidebar.")
    st.stop()

# ---- uploads ----
uploaded = st.file_uploader(
    "Drop page image(s) or a .zip of pages here",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "zip"],
    accept_multiple_files=True)

# unpack uploads into {name: bytes}
inputs = {}
if uploaded:
    for f in uploaded:
        name = f.name
        data = f.getvalue()
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith(SUPPORTED_EXTS) \
                            and not info.is_dir():
                        inputs[info.filename] = zf.read(info.filename)
        elif name.lower().endswith(SUPPORTED_EXTS):
            inputs[name] = data

if not inputs:
    st.info("👆 Upload one or more images, or a ZIP archive of images "
            "(PNG / JPG / WebP / BMP / TIFF).")
    st.stop()

st.success(f"{len(inputs)} image(s) ready: "
           + ", ".join(sorted(inputs)[:5])
           + (" …" if len(inputs) > 5 else ""))

run = st.button("✨ Remove text", type="primary", use_container_width=True)

if run:
    reader = get_reader(tuple(languages), gpu)
    lama = get_lama("cuda" if gpu else "cpu")

    results = {}   # name -> {"clean": bytes, "mask": bytes|None}
    prog = st.progress(0.0, text="Starting …")
    status = st.empty()

    for i, (name, data) in enumerate(sorted(inputs.items())):
        status.markdown(f"**Processing** `{name}` "
                        f"({i + 1}/{len(inputs)})")
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not decode image")
            h, w = img.shape[:2]

            boxes = detect_text_boxes(img, reader, int(win),
                                      int(overlap), float(conf))
            mask = build_mask(img.shape, boxes, int(dilate))
            clean = inpaint_lama_tiled(img, mask, lama,
                                       tile=int(tile))

            entry = {"clean": cv2.imencode(".png", clean)[1].tobytes(),
                     "mask": (cv2.imencode(".png", mask)[1].tobytes()
                              if save_mask else None),
                     "size": f"{w}×{h}", "boxes": len(boxes)}
        except Exception as e:                       # noqa: BLE001
            entry = {"error": str(e)}
        results[name] = entry
        prog.progress((i + 1) / len(inputs),
                      text=f"Done {i + 1}/{len(inputs)}")
    status.empty()
    st.session_state["results"] = results
    st.success("All done! Scroll down to download. ✨")

# ---- results / downloads ----
if "results" in st.session_state:
    results = st.session_state["results"]
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, entry in sorted(results.items()):
            if "error" in entry:
                continue
            base = name.rsplit(".", 1)[0]
            zf.writestr(f"clean_{base}.png", entry["clean"])
            if entry.get("mask"):
                zf.writestr(f"mask_{base}.png", entry["mask"])
    zip_buf.seek(0)

    ok = [n for n, e in results.items() if "error" not in e]
    bad = [n for n, e in results.items() if "error" in e]
    c1, c2 = st.columns(2)
    c1.metric("Cleaned", len(ok))
    c2.metric("Failed", len(bad))

    st.download_button("⬇️ Download everything (.zip)",
                       data=zip_buf.getvalue(),
                       file_name="cleaned_results.zip",
                       mime="application/zip",
                       use_container_width=True)

    for name, entry in sorted(results.items()):
        with st.expander(f"{'✅' if 'error' not in entry else '❌'} {name}"):
            if "error" in entry:
                st.error(entry["error"])
                continue
            st.caption(f"{entry['size']} · {entry['boxes']} text "
                       f"region(s) removed")
            col_a, col_b = st.columns(2)
            orig_pil = Image.open(io.BytesIO(inputs[name]))
            clean_pil = Image.open(io.BytesIO(entry["clean"]))
            col_a.image(orig_pil, caption="Original", use_container_width=True)
            col_b.image(clean_pil, caption="Cleaned", use_container_width=True)
            base = name.rsplit(".", 1)[0]
            col_a.download_button("⬇️ clean PNG",
                                  entry["clean"],
                                  f"clean_{base}.png",
                                  "image/png")
            if entry.get("mask"):
                col_b.download_button("⬇️ mask PNG",
                                      entry["mask"],
                                      f"mask_{base}.png",
                                      "image/png")

    if bad:
        st.warning("Some files failed: " + ", ".join(bad))
