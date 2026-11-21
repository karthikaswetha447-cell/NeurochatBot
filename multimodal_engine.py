"""NeuroBot multimodal helpers: Groq vision, speech transcription/TTS and video frames.

The existing GPT-OSS text model remains the default text chatbot. Vision and audio use
separate model IDs because model capabilities differ by modality.
"""
import base64
import mimetypes
import os
import tempfile
import shutil
import subprocess
from pathlib import Path

from groq import Groq

VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "troy")
MAX_UPLOAD_MB = int(os.getenv("MULTIMODAL_MAX_UPLOAD_MB", "20"))
MAX_VIDEO_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "6"))


def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured in Render.")
    return Groq(api_key=key)


def _check_file(path, allowed, max_mb=MAX_UPLOAD_MB):
    if not path or not os.path.isfile(path):
        raise ValueError("Please upload a valid file.")
    size = os.path.getsize(path)
    if size > max_mb * 1024 * 1024:
        raise ValueError(f"File is too large. Maximum size is {max_mb} MB.")
    suffix = Path(path).suffix.lower()
    if suffix not in allowed:
        raise ValueError("Unsupported file type: " + (suffix or "unknown"))
    return suffix


def _image_data_url(path):
    # Leave headroom for base64 encoding within the provider request-size limit.
    _check_file(path, {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}, max_mb=14)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def analyze_image(path, question, history=None):
    """Return vision answer. History may contain plain text turns."""
    image_url = _image_data_url(path)
    messages = [{"role": "system", "content": "You are NeuroBot. Carefully analyze the supplied image. Describe visible text, diagrams, charts and relevant details. Do not guess unreadable text."}]
    for turn in (history or [])[-6:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": question or "Describe this image and explain its important details."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]})
    response = _client().chat.completions.create(
        model=VISION_MODEL, messages=messages, temperature=0.2,
        max_completion_tokens=1200, top_p=1
    )
    return response.choices[0].message.content or "I could not analyze this image."


def transcribe_audio(path, language=None):
    _check_file(path, {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"})
    with open(path, "rb") as f:
        result = _client().audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model=TRANSCRIPTION_MODEL,
            response_format="json",
            temperature=0,
            **({"language": language} if language else {})
        )
    return result.text


def generate_speech(text, output_path=None):
    """Groq Orpheus TTS currently supports English. Returns an MP3 path."""
    text = (text or "").strip()
    if not text:
        raise ValueError("There is no text to speak.")
    if len(text) > 4000:
        text = text[:4000]
    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="neurobot_voice_", suffix=".wav")
        os.close(fd)
    response = _client().audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav"
    )
    response.write_to_file(output_path)
    return output_path


def _extract_video_frames(path, max_frames=MAX_VIDEO_FRAMES):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video processing dependency OpenCV is not installed.") from exc
    _check_file(path, {".mp4", ".mov", ".avi", ".mkv", ".webm"}, max_mb=100)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open this video file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError("No readable video frames were found.")
    positions = sorted(set(int(i * max(total - 1, 0) / max(max_frames - 1, 1)) for i in range(min(max_frames, total))))
    paths = []
    temp_dir = tempfile.mkdtemp(prefix="neurobot_frames_")
    for n, frame_no in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = os.path.join(temp_dir, f"frame_{n}.jpg")
        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        paths.append(frame_path)
    cap.release()
    if not paths:
        raise ValueError("Could not extract video frames.")
    return paths, temp_dir


def analyze_video(path, question):
    """Practical video understanding via sampled frames. Audio track is not analyzed here."""
    frames, temp_dir = _extract_video_frames(path)
    try:
        frame_urls = [_image_data_url(p) for p in frames[:3]]
        frame_urls2 = [_image_data_url(p) for p in frames[3:6]]
        answers = []
        for group_no, group in enumerate([frame_urls, frame_urls2], start=1):
            if not group:
                continue
            content = [{"type": "text", "text": (
                f"Analyze these sampled frames from video section {group_no}. "
                + (question or "Summarize the visible events and sequence.")
                + " Mention that these are sampled frames, not continuous video.")
            }]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in group)
            response = _client().chat.completions.create(
                model=VISION_MODEL, messages=[{"role": "user", "content": content}],
                temperature=0.2, max_completion_tokens=900, top_p=1
            )
            answers.append(response.choices[0].message.content or "")
        video_audio_note = "Video audio: not extracted."
        try:
            import imageio_ffmpeg
            audio_path = os.path.join(temp_dir, "video_audio.wav")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", audio_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=90
            )
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                video_transcript = transcribe_audio(audio_path)
                video_audio_note = "VIDEO AUDIO TRANSCRIPTION\n" + video_transcript
            else:
                video_audio_note = "Video contains no extractable audio track."
        except Exception as audio_error:
            video_audio_note = "Video audio could not be extracted/transcribed: " + str(audio_error)[:180]
        return ("VIDEO FRAME ANALYSIS (sampled frames; not full-motion understanding)\n\n"
                + "\n\n".join(answers) + "\n\n" + video_audio_note)
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def analyze_multimodal(image_path=None, audio_path=None, video_path=None, question=""):
    """Single Gradio callback entry point. Returns (result, transcription)."""
    outputs, transcript = [], ""
    if image_path:
        outputs.append("IMAGE ANALYSIS\n" + analyze_image(image_path, question))
    if audio_path:
        transcript = transcribe_audio(audio_path)
        outputs.append("AUDIO TRANSCRIPTION\n" + transcript)
    if video_path:
        outputs.append(analyze_video(video_path, question))
    if not outputs:
        raise ValueError("Upload an image, audio file or video first.")
    return "\n\n---\n\n".join(outputs), transcript
