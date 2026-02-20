import functools
import os
import tempfile

import gradio as gr
import yt_dlp
from faster_whisper import WhisperModel

# ── Model caching ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def load_model(size: str) -> WhisperModel:
    """Load and cache a WhisperModel (one instance per model size)."""
    return WhisperModel(size, device="cpu", compute_type="int8")


# ── YouTube download ──────────────────────────────────────────────────────────
def download_youtube_audio(url: str) -> tuple[str, str]:
    """
    Download audio from a YouTube URL to a temp file.
    Returns (file_path, video_title).
    """
    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")

    # Find the downloaded mp3 file
    for f in os.listdir(tmp_dir):
        if f.endswith(".mp3"):
            return os.path.join(tmp_dir, f), title

    raise FileNotFoundError("Le fichier audio téléchargé est introuvable.")


# ── Transcription function ────────────────────────────────────────────────────
def _run_transcription(audio_path: str, model_size: str) -> tuple[str, str]:
    """Core transcription logic, shared by both tabs."""
    model = load_model(model_size)

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=True,
    )

    info_text = (
        f"✅ Langue détectée : **{info.language}** "
        f"(confiance : {info.language_probability:.1%})"
    )

    lines = [
        "[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text)
        for seg in segments
    ]
    return info_text, "\n".join(lines)


def transcribe_file(audio_path: str, model_size: str) -> tuple[str, str]:
    if audio_path is None:
        return "", "⚠️ Aucun fichier audio fourni."
    return _run_transcription(audio_path, model_size)


def transcribe_youtube(url: str, model_size: str) -> tuple[str, str]:
    if not url or not url.strip():
        return "", "⚠️ Veuillez entrer une URL YouTube."
    try:
        audio_path, title = download_youtube_audio(url.strip())
    except Exception as e:
        return "", f"❌ Erreur lors du téléchargement : {e}"
    try:
        info_text, transcription = _run_transcription(audio_path, model_size)
        info_text = f"🎬 **{title}**\n\n{info_text}"
        return info_text, transcription
    finally:
        os.unlink(audio_path)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🎙️ Whisper Transcription") as demo:
    gr.Markdown(
        "# 🎙️ Whisper Transcription\n"
        "Transcription audio automatique avec "
        "[faster-whisper](https://github.com/SYSTRAN/faster-whisper)"
    )

    model_selector = gr.Dropdown(
        choices=["small", "medium", "large-v3", "distil-large-v3"],
        value="small",
        label="Modèle Whisper",
        info=(
            "small : rapide | medium : compromis | "
            "large-v3 : meilleure qualité | distil-large-v3 : distillé"
        ),
    )

    with gr.Tabs():

        # ── Tab 1 : Fichier / Microphone ──────────────────────────────────────
        with gr.Tab("📂 Fichier / Micro"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Fichier audio",
                    )
                    file_btn = gr.Button("🚀 Lancer la transcription", variant="primary")
                with gr.Column(scale=2):
                    file_info = gr.Markdown(label="Informations")
                    file_output = gr.Textbox(label="Transcription", lines=20)

            file_btn.click(
                fn=transcribe_file,
                inputs=[audio_input, model_selector],
                outputs=[file_info, file_output],
            )

        # ── Tab 2 : YouTube ───────────────────────────────────────────────────
        with gr.Tab("▶️ YouTube"):
            with gr.Row():
                with gr.Column(scale=1):
                    yt_url = gr.Textbox(
                        label="URL YouTube",
                        placeholder="https://www.youtube.com/watch?v=...",
                    )
                    yt_btn = gr.Button("🚀 Télécharger & Transcrire", variant="primary")
                with gr.Column(scale=2):
                    yt_info = gr.Markdown(label="Informations")
                    yt_output = gr.Textbox(label="Transcription", lines=20)

            yt_btn.click(
                fn=transcribe_youtube,
                inputs=[yt_url, model_selector],
                outputs=[yt_info, yt_output],
            )

demo.launch(server_name="0.0.0.0", server_port=7860)