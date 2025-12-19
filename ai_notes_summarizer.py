import streamlit as st
# NOTE: transformers (and torch) can pull in native DLLs on import which may
# fail on Windows if Visual C++ redistributable is missing. Import transformers
# lazily inside `load_local_model` so the module import doesn't crash the app.
from typing import Optional
try:
    from huggingface_hub import InferenceClient
    HAS_HF_HUB = True
except Exception:
    InferenceClient = None
    HAS_HF_HUB = False
from PIL import Image
import pdfplumber
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="📘",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_local_model(model_name: str = "facebook/bart-large-cnn"):
    """Load and cache a local transformers summarization pipeline."""
    try:
        # import lazily to avoid importing torch at module import time
        from transformers import pipeline
    except Exception as e:
        # Provide a helpful error message about why local models are unavailable
        # and raise a RuntimeError the UI can catch.
        raise RuntimeError(
            "Local model support is unavailable because importing `transformers` failed. "
            "This commonly happens on Windows when the Microsoft Visual C++ Redistributable is not installed, or when PyTorch is not compatible with this Python build. "
            "Install the redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) and ensure a compatible PyTorch is installed, or switch to the Hugging Face Inference API backend.") from e

    return pipeline(
        "summarization",
        model=model_name
    )


def load_hf_client(token: Optional[str]):
    """Return an InferenceClient for the HF Inference API. Token may be None for anonymous requests."""
    if not HAS_HF_HUB or InferenceClient is None:
        raise RuntimeError("huggingface_hub is not installed. Install it with `pip install huggingface-hub` to use hf_api backend.")
    if not token:
        return InferenceClient()
    return InferenceClient(token=token)

# ---------------- FUNCTIONS ----------------
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """Split text into chunks of approximately `chunk_size` characters with `overlap`."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= chunk_size:
            current = current + "\n" + p if current else p
        else:
            chunks.append(current)
            # start new chunk with overlap
            current = p[-overlap:] if overlap < len(p) else p
    if current:
        chunks.append(current)
    return chunks


@st.cache_data
def summarize_chunk_local(model_name: str, text: str, max_len: int, min_len: int):
    model = load_local_model(model_name)
    out = model(text, max_length=max_len, min_length=min_len, do_sample=False)
    return out[0]["summary_text"]


def summarize_chunk_hf(client, model_name: str, text: str, max_len: int, min_len: int):
    # The HF Inference API expects the model and payload
    resp = client.text_summarization(model=model_name, inputs=text, max_new_tokens=max_len)
    # resp can be a dict or str depending on client version
    if isinstance(resp, dict) and "summary_text" in resp:
        return resp["summary_text"]
    if isinstance(resp, str):
        return resp
    # fallback
    return str(resp)


def summarize_text(text: str, *, backend: str, model_name: str, max_len: int, min_len: int, hf_client: Optional[object] = None, chunk_size: int = 2000, overlap: int = 200):
    """Summarize text with chunking + map-reduce when necessary.
    backend: 'local' or 'hf_api'
    If the document is split into multiple chunks we summarize each chunk then summarize the joined summaries.
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # progress helper
    summaries = []
    for i, c in enumerate(chunks, start=1):
        if backend == "local":
            s = summarize_chunk_local(model_name, c, max_len, min_len)
        else:
            if not hf_client:
                raise RuntimeError("Hugging Face client required for hf_api backend")
            s = summarize_chunk_hf(hf_client, model_name, c, max_len, min_len)
        summaries.append(s)

    if len(summaries) == 1:
        return summaries[0]

    # reduce step: join chunk summaries and summarize again to make concise
    joined = "\n\n".join(summaries)
    if backend == "local":
        return summarize_chunk_local(model_name, joined, max_len, min_len)
    else:
        return summarize_chunk_hf(hf_client, model_name, joined, max_len, min_len)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def _find_tesseract_executable() -> str | None:
    """Return the path to the tesseract executable if available, otherwise None."""
    # Check common places via shutil.which
    path = shutil.which("tesseract")
    if path:
        return path
    # On Windows, tesseract may be installed in Program Files (UB Mannheim builds)
    possible = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    return None


def is_ocr_available() -> bool:
    """Return True if both pytesseract and tesseract binary are available."""
    try:
        import pytesseract  # lazy import
    except Exception:
        return False
    return _find_tesseract_executable() is not None


def extract_text_from_image(image):
    """Extract text from a PIL Image using pytesseract.

    This function imports pytesseract lazily and checks for the tesseract binary.
    Raises RuntimeError with actionable instructions if requirements are not met.
    """
    try:
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "The Python package `pytesseract` is not installed. Install it with `pip install pytesseract` and ensure the Tesseract OCR binary is installed on your system."
        ) from e

    tesseract_path = _find_tesseract_executable()
    if not tesseract_path:
        raise RuntimeError(
            "Tesseract binary not found. On Windows, install it from https://github.com/UB-Mannheim/tesseract/wiki and ensure the installation folder (e.g., C:\\Program Files\\Tesseract-OCR) is on your PATH."
        )

    # Ensure pytesseract uses the discovered binary
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return pytesseract.image_to_string(image)

def save_summary_pdf(summary_text):
    file_path = "summary.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    y = height - 40

    for line in summary_text.split("\n"):
        c.drawString(40, y, line)
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40

    c.save()
    return file_path

def save_summary_image(summary_text):
    img = Image.new("RGB", (900, 600), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    x, y = 20, 20
    for line in summary_text.split(". "):
        draw.text((x, y), line, fill="black")
        y += 25

    img_path = "summary.png"
    img.save(img_path)
    return img_path

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Settings")

# Backend selection
backend = st.sidebar.radio("Backend:", ["local", "hf_api"], index=0)

# Model selection
local_models = ["facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6", "google/flan-t5-large"]
hf_models = ["facebook/bart-large-cnn", "google/flan-t5-large", "sshleifer/distilbart-cnn-12-6"]
model_name = st.sidebar.selectbox("Model:", local_models if backend == "local" else hf_models, index=0)

# HF API key
hf_token = None
if backend == "hf_api":
    hf_token = st.sidebar.text_input("Hugging Face token (optional)", type="password")

max_len = st.sidebar.slider("Max Summary Length", 80, 1024, 150)
min_len = st.sidebar.slider("Min Summary Length", 30, 300, 60)

# Chunking options
chunk_size = st.sidebar.slider("Chunk size (chars)", 500, 8000, 2000)
overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 1000, 200)

st.sidebar.markdown("---")
st.sidebar.markdown("**Mode:** Map-Reduce chunking is used for long inputs to avoid token limits and improve quality.")

# ---------------- MAIN UI ----------------
st.title("📘 AI Notes Summarizer App")
st.write(
    "Upload **PDF / Image notes** or **Paste text** to get summarized notes."
)

# Input selection (include image again now that OCR is opt-in)
input_type = st.radio(
    "Select Input Type:",
    ["Paste Text", "Upload PDF", "Upload Image"]
)

# Opt-in OCR setting (disabled by default)
ocr_enabled = st.sidebar.checkbox("Enable OCR (requires Tesseract)", value=False)
ocr_available = is_ocr_available()
st.sidebar.markdown(f"**OCR available:** {'✅ Yes' if ocr_available else '❌ No'}")
if not ocr_available:
    st.sidebar.caption("If you enable OCR but prerequisites are missing, the app will skip OCR and continue. Install Tesseract (Windows: UB Mannheim) and `pytesseract` to enable it.")
else:
    st.sidebar.caption("Requires `pytesseract` and the Tesseract binary installed on your system. On Windows: https://github.com/UB-Mannheim/tesseract/wiki")

extracted_text = ""

# ---------------- INPUT HANDLING ----------------
if input_type == "Paste Text":
    extracted_text = st.text_area(
        "Paste your notes here:",
        height=300
    )

elif input_type == "Upload PDF":
    pdf_file = st.file_uploader("Upload PDF Notes", type=["pdf"])
    if pdf_file:
        extracted_text = extract_text_from_pdf(pdf_file)
        st.success("✅ Text extracted from PDF")

elif input_type == "Upload Image":
    image_file = st.file_uploader("Upload Image Notes", type=["png", "jpg", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if not ocr_enabled:
            st.warning("OCR is disabled. Enable 'Enable OCR' in the sidebar to extract text from images.")
        else:
            if not ocr_available:
                # If prerequisites missing, don't attempt OCR — just warn and continue
                st.warning("OCR prerequisites are missing (Tesseract or pytesseract). Image text will not be extracted. You can paste text or upload a PDF instead.")
            else:
                with st.spinner("Extracting text from image..."):
                    try:
                        extracted_text = extract_text_from_image(image)
                        st.success("✅ Text extracted from Image")
                    except Exception as e:
                        # Non-blocking warning — leave OCR disabled behavior graceful
                        st.warning(f"OCR failed, skipping extraction: {e}")

# ---------------- SUMMARIZE ----------------
if st.button("🚀 Generate Summary"):
    if len(extracted_text.strip()) < 50:
        st.warning("⚠ Please provide more content for summarization (try >50 chars).")
    else:
        st.subheader("Summarization progress")
        progress_bar = st.progress(0)
        status = st.empty()

        try:
            status.info("Preparing backend...")
            hf_client = None
            if backend == "local":
                # Try loading the local model; if it fails due to missing DLLs
                # or incompatible torch, switch to hf_api backend and notify the user.
                try:
                    _ = load_local_model(model_name)
                except RuntimeError as e:
                    # Inform the user and fall back to hf_api when available
                    st.error(str(e))
                    if HAS_HF_HUB:
                        st.warning("Falling back to Hugging Face Inference API backend.")
                        backend = "hf_api"
                        try:
                            hf_client = load_hf_client(hf_token)
                        except Exception as e2:
                            st.error(f"Failed to initialize Hugging Face client: {e2}")
                            hf_client = None
                    else:
                        st.error("Hugging Face Hub SDK not available; please install `huggingface-hub` or fix local model dependencies.")
                        raise
            else:
                try:
                    hf_client = load_hf_client(hf_token)
                except Exception as e:
                    st.error(f"Failed to initialize Hugging Face client: {e}")
                    hf_client = None

            status.info("Chunking document...")
            chunks = chunk_text(extracted_text, chunk_size=chunk_size, overlap=overlap)
            total = len(chunks)

            # Summarize each chunk and update progress
            chunk_summaries = []
            for i, c in enumerate(chunks, start=1):
                status.info(f"Summarizing chunk {i}/{total}...")
                if backend == "local":
                    s = summarize_chunk_local(model_name, c, max_len, min_len)
                else:
                    s = summarize_chunk_hf(hf_client, model_name, c, max_len, min_len)
                chunk_summaries.append(s)
                progress_bar.progress(int(i / total * 100))

            if total > 1:
                status.info("Reducing chunk summaries...")
                joined = "\n\n".join(chunk_summaries)
                if backend == "local":
                    summary_text = summarize_chunk_local(model_name, joined, max_len, min_len)
                else:
                    summary_text = summarize_chunk_hf(hf_client, model_name, joined, max_len, min_len)
            else:
                summary_text = chunk_summaries[0]

            progress_bar.progress(100)
            status.success("Done")

            st.subheader("✂ Summarized Notes")
            st.text_area("Summary", summary_text, height=250)

            # ---------------- DOWNLOAD OPTIONS ----------------
            st.subheader("⬇ Download Summary")

            # Text download
            st.download_button(
                "📄 Download as Text",
                summary_text,
                file_name="summary.txt"
            )

            # PDF download
            pdf_path = save_summary_pdf(summary_text)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📕 Download as PDF",
                    f,
                    file_name="summary.pdf"
                )

            # Image download
            img_path = save_summary_image(summary_text)
            with open(img_path, "rb") as f:
                st.download_button(
                    "🖼 Download as Image",
                    f,
                    file_name="summary.png"
                )

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ — powered by Transformers and Hugging Face By Nehal Ahmed.")

    