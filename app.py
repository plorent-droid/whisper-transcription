import functools

import gradio as gr
from faster_whisper import WhisperModel

# ── Model caching ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def load_model(size: str) -> WhisperModel:
    """Load and cache a WhisperModel (one instance per model size)."""
    return WhisperModel(size, device="cpu", compute_type="int8")


# ── Transcription function ────────────────────────────────────────────────────
def transcribe(audio_path: str, model_size: str) -> tuple[str, str]:
    """
    Transcribe an audio file and return (info_text, transcription_text).
    `audio_path` is a filepath provided by Gradio's Audio component.
    """
    if audio_path is None:
        return "", "⚠️ Aucun fichier audio fourni."

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
    transcription = "\n".join(lines)

    return info_text, transcription


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🎙️ Whisper Transcription") as demo:
    gr.Markdown(
        "# 🎙️ Whisper Transcription\n"
        "Transcription audio automatique avec "
        "[faster-whisper](https://github.com/SYSTRAN/faster-whisper)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=["small", "medium", "large-v3", "distil-large-v3"],
                value="small",
                label="Modèle Whisper",
                info=(
                    "small : rapide | medium : compromis | "
                    "large-v3 : meilleure qualité | distil-large-v3 : distillé"
                ),
            )
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Fichier audio",
            )
            run_btn = gr.Button("🚀 Lancer la transcription", variant="primary")

        with gr.Column(scale=2):
            info_output = gr.Markdown(label="Informations")
            text_output = gr.Textbox(
                label="Transcription",
                lines=20,
                show_copy_button=True,
            )

    run_btn.click(
        fn=transcribe,
        inputs=[audio_input, model_selector],
        outputs=[info_output, text_output],
    )

demo.launch()