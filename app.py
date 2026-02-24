import functools
import os
import tempfile
import urllib.request

import gradio as gr
from faster_whisper import WhisperModel

# ── Model caching ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def load_model(size: str) -> WhisperModel:
    """Load and cache a WhisperModel (one instance per model size)."""
    return WhisperModel(size, device="cpu", compute_type="int8")


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


def transcribe_from_url(url: str, model_size: str) -> tuple[str, str]:
    """Download audio from a direct URL (not YouTube) and transcribe it."""
    if not url or not url.strip():
        return "", "⚠️ Veuillez entrer une URL."

    url = url.strip()

    # Bloquer explicitement YouTube
    blocked = ["youtube.com", "youtu.be"]
    if any(b in url for b in blocked):
        return "", (
            "❌ Les URLs YouTube sont bloquées sur HuggingFace Spaces.\n\n"
            "**Alternative :** Téléchargez la vidéo/audio sur votre ordinateur "
            "avec [yt-dlp](https://github.com/yt-dlp/yt-dlp) puis uploadez le fichier dans l'onglet 📂 Fichier / Micro.\n\n"
            "```bash\nyt-dlp -x --audio-format mp3 \"URL_YOUTUBE\"\n```"
        )

    try:
        tmp_dir = tempfile.mkdtemp()
        ext = url.split("?")[0].split(".")[-1] or "mp3"
        tmp_path = os.path.join(tmp_dir, f"audio.{ext}")

        urllib.request.urlretrieve(url, tmp_path)

    except Exception as e:
        return "", f"❌ Erreur lors du téléchargement : {e}"

    try:
        info_text, transcription = _run_transcription(tmp_path, model_size)
        return info_text, transcription
    finally:
        os.unlink(tmp_path)


def prepare_download(transcription: str) -> str | None:
    """Write transcription text to a temp .txt file and return its path."""
    if not transcription or not transcription.strip():
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(transcription)
    tmp.close()
    return tmp.name


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
                    file_dl_btn = gr.Button("⬇️ Télécharger la transcription")
                    file_dl = gr.File(label="Fichier à télécharger", visible=False)

            file_btn.click(
                fn=transcribe_file,
                inputs=[audio_input, model_selector],
                outputs=[file_info, file_output],
            )
            file_dl_btn.click(
                fn=prepare_download,
                inputs=[file_output],
                outputs=[file_dl],
            ).then(fn=lambda f: gr.File(visible=f is not None), inputs=[file_dl], outputs=[file_dl])

        # ── Tab 2 : URL directe (YouTube remplacé) ────────────────────────────
        with gr.Tab("🔗 URL Audio"):
            gr.Markdown(
                "> ⚠️ **YouTube est bloqué** sur HuggingFace Spaces (restriction réseau).\n"
                "> Utilisez une URL directe vers un fichier audio (`.mp3`, `.wav`, `.m4a`…).\n"
                "> Pour YouTube, téléchargez d'abord l'audio localement :\n"
                "> ```bash\n> yt-dlp -x --audio-format mp3 \"URL_YOUTUBE\"\n> ```\n"
                "> puis uploadez le fichier dans l'onglet **📂 Fichier / Micro**."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    url_input = gr.Textbox(
                        label="URL directe vers un fichier audio",
                        placeholder="https://example.com/audio.mp3",
                    )
                    url_btn = gr.Button("🚀 Télécharger & Transcrire", variant="primary")
                with gr.Column(scale=2):
                    url_info = gr.Markdown(label="Informations")
                    url_output = gr.Textbox(label="Transcription", lines=20)
                    url_dl_btn = gr.Button("⬇️ Télécharger la transcription")
                    url_dl = gr.File(label="Fichier à télécharger", visible=False)

            url_btn.click(
                fn=transcribe_from_url,
                inputs=[url_input, model_selector],
                outputs=[url_info, url_output],
            )
            url_dl_btn.click(
                fn=prepare_download,
                inputs=[url_output],
                outputs=[url_dl],
            ).then(fn=lambda f: gr.File(visible=f is not None), inputs=[url_dl], outputs=[url_dl])

demo.launch(server_name="0.0.0.0", server_port=7860)