import os
import sys
import re
import math
import time
import shutil
import tempfile
import subprocess
import threading
import uuid
import inspect
import json
import hashlib
import html
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import scipy.io.wavfile as wavfile
from datetime import datetime

# --- Paths (must be resolved BEFORE importing gradio) -----------------------
# gradio.route_utils reads GRADIO_TEMP_DIR at import time. The default
# %TEMP%\gradio\<hash>\<original name> is the spot Windows locks right after an
# upload, which produced "PermissionError: [Errno 13]" in starlette.stream_file
# when the browser tried to stream a freshly uploaded Cyrillic-named video.
# Keeping uploads inside the project also makes the path short and easy to clean.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GRADIO_TEMP_DIR = os.path.join(CURRENT_DIR, "gradio_tmp")
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

# Analysis cache lives in the project next to the code, so a re-run with other render
# settings never has to redo Whisper / RMS / motion analysis.
CACHE_DIR = os.path.join(CURRENT_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_VERSION = 1
UPLOAD_MAX_AGE_HOURS = 24

import gradio as gr  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

# Path to FFmpeg executable
LOCAL_FFMPEG = os.path.join(CURRENT_DIR, "ffmpeg.exe")
FFMPEG_BIN = LOCAL_FFMPEG if os.path.isfile(LOCAL_FFMPEG) else "ffmpeg"

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

# Global FFmpeg flags: keep stderr tiny and never wait on stdin.
# Without them FFmpeg writes ~32 KB of progress per 60 s clip into a Windows pipe
# buffer of ~4 KB, blocks on the write and the render freezes forever.
FFMPEG_QUIET_FLAGS = ["-nostdin", "-nostats"]

# Keyframe-only decode for the motion pass: ~9x faster on 1080p60 (measured on this
# machine), at the cost of a coarser grid (keyframe interval, ~6 s here). The diffs are
# placed on the 1 Hz grid by their real showinfo timestamps, so scoring stays correct.
# Set to False for a 1 fps scan of every frame.
MOTION_KEYFRAMES_ONLY = True

# Flags required for a browser / social-media friendly MP4
OUTPUT_VIDEO_FLAGS = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def _popen_ffmpeg(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, log_level="error"):
    """Starts FFmpeg with the quiet flags injected and a hidden console window on Windows.

    log_level="info" is used where frame timestamps are needed (showinfo).
    """
    full_cmd = [cmd[0]] + FFMPEG_QUIET_FLAGS + ["-loglevel", log_level] + list(cmd[1:])
    kwargs = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return subprocess.Popen(full_cmd, stdout=stdout, stderr=stderr, **kwargs)


def _drain_pipe(stream, sink):
    """Reads a pipe to EOF in a background thread so FFmpeg can never block on a full buffer."""
    try:
        data = stream.read()
        if data:
            sink.append(data)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _terminate_proc(proc):
    """Stops a running FFmpeg process, escalating to kill if it ignores terminate."""
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        pass


def _read_stdout_cancellable(proc, chunk_size):
    """Reads proc.stdout in chunks and stops at once if STOP_EVENT is set (caller kills FFmpeg)."""
    chunks = []
    while True:
        if STOP_EVENT.is_set():
            break
        buf = proc.stdout.read(chunk_size)
        if not buf:
            break
        chunks.append(buf)
    try:
        proc.stdout.close()
    except Exception:
        pass
    return b"".join(chunks)

HOOK_WORDS = {
    "как", "почему", "зачем", "что", "где", "когда", "кто", "сколько",
    "секрет", "ошибка", "правда", "никогда", "главное", "топ", "смотри",
    "знаешь", "важно", "невероятно", "лайфхак", "факт", "проблема", "причина",
    "хитрость", "смысл", "понять", "выяснить", "внимание", "запомни"
}

FORMAT_RESOLUTIONS = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080)
}

# Global cancellation event for stopping processing
STOP_EVENT = threading.Event()


def stop_processing():
    """Sets the global stop event flag to interrupt processing immediately."""
    STOP_EVENT.set()
    return "🛑 Бурмалда остановлена пользователем."


def run_subprocess_cancellable(cmd, timeout_seconds=3600):
    """Runs FFmpeg with cancellation support, a drained stderr pipe and a hard timeout.

    stderr is consumed by a background thread and kept small by -nostats/-loglevel error,
    so the small Windows pipe buffer can never fill up and block the encoder.
    """
    proc = _popen_ffmpeg(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err_chunks = []
    drainer = threading.Thread(target=_drain_pipe, args=(proc.stderr, err_chunks), daemon=True)
    drainer.start()

    started_at = time.time()
    while proc.poll() is None:
        if STOP_EVENT.is_set():
            _terminate_proc(proc)
            raise RuntimeError("🛑 Бурмалда остановлена пользователем.")
        if timeout_seconds and (time.time() - started_at) > timeout_seconds:
            _terminate_proc(proc)
            raise RuntimeError(
                f"FFmpeg не завершился за {int(timeout_seconds)} с и был остановлен. "
                f"Проверьте исходный файл или уменьшите длину клипа."
            )
        time.sleep(0.15)

    drainer.join(timeout=2)

    if STOP_EVENT.is_set():
        raise RuntimeError("🛑 Бурмалда остановлена пользователем.")

    stderr_text = b"".join(err_chunks).decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg execution failed: {stderr_text}")
    return stderr_text


def _parse_progress_seconds(line: str) -> float:
    """Parses one -progress line (out_time_us / out_time_ms / out_time) into seconds, -1 if not a time line."""
    if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
        # both fields carry microseconds in FFmpeg's -progress output
        value = line.split("=", 1)[1].strip()
        if value and value != "N/A":
            try:
                return int(value) / 1_000_000.0
            except ValueError:
                return -1.0
        return -1.0
    if line.startswith("out_time="):
        value = line.split("=", 1)[1].strip()
        if value and value != "N/A":
            try:
                hours, minutes, seconds = value.split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            except ValueError:
                return -1.0
    return -1.0


def run_ffmpeg_with_progress(cmd, duration_seconds, progress_cb=None, timeout_seconds=3600):
    """Runs FFmpeg and reports real encode progress parsed from `-progress pipe:1`.

    progress_cb receives 0.0..1.0. stdout and stderr are each drained by their own
    thread, so neither pipe can block the encoder on Windows.
    """
    proc = _popen_ffmpeg(list(cmd) + ["-progress", "pipe:1"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err_chunks = []
    threading.Thread(target=_drain_pipe, args=(proc.stderr, err_chunks), daemon=True).start()

    state = {"seconds": 0.0}

    def _read_progress():
        try:
            for raw_line in iter(proc.stdout.readline, b""):
                seconds = _parse_progress_seconds(raw_line.decode("utf-8", "replace").strip())
                if seconds >= state["seconds"]:
                    state["seconds"] = seconds
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    threading.Thread(target=_read_progress, daemon=True).start()

    started_at = time.time()
    last_report = 0.0
    while proc.poll() is None:
        if STOP_EVENT.is_set():
            _terminate_proc(proc)
            raise RuntimeError("🛑 Бурмалда остановлена пользователем.")
        if timeout_seconds and (time.time() - started_at) > timeout_seconds:
            _terminate_proc(proc)
            raise RuntimeError(
                f"FFmpeg не завершился за {int(timeout_seconds)} с и был остановлен. "
                f"Проверьте исходный файл или уменьшите длину клипа."
            )
        if progress_cb is not None and duration_seconds and (time.time() - last_report) > 0.4:
            fraction = max(0.0, min(1.0, state["seconds"] / float(duration_seconds)))
            try:
                progress_cb(fraction)
            except Exception:
                pass
            last_report = time.time()
        time.sleep(0.1)

    if progress_cb is not None:
        try:
            progress_cb(1.0)
        except Exception:
            pass

    stderr_text = b"".join(err_chunks).decode("utf-8", "replace").strip()
    if STOP_EVENT.is_set():
        raise RuntimeError("🛑 Бурмалда остановлена пользователем.")
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg execution failed: {stderr_text}")
    return stderr_text


# --- Analysis cache --------------------------------------------------------

def source_fingerprint(path: str) -> str:
    """Content identity of a source file: size plus a hash of its head and tail.

    Hashing a 2 GB video completely would cost more than the analysis it protects, and
    mtime is deliberately NOT used: Gradio re-copies an upload on every run, so the file
    would never match otherwise. Head + tail + exact size is enough in practice.
    """
    try:
        stat = os.stat(path)
        digest = hashlib.sha1()
        digest.update(str(stat.st_size).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if stat.st_size > 2 * 1024 * 1024:
                handle.seek(-1024 * 1024, os.SEEK_END)
                digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()
    except OSError:
        return ""


def analysis_cache_path(fingerprint: str, start: float, duration) -> str:
    if not fingerprint:
        return ""
    span = "full" if not duration else f"{float(start):.3f}_{float(duration):.3f}"
    return os.path.join(CACHE_DIR, f"{fingerprint}_{span}.json")


def load_analysis_cache(fingerprint: str, start: float, duration):
    """Returns (words, segments, rms_per_sec, motion_per_sec, media_duration) or None."""
    path = analysis_cache_path(fingerprint, start, duration)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("version") != CACHE_VERSION:
            return None
        return (
            data["words"],
            data["segments"],
            np.asarray(data["rms"], dtype=np.float32),
            np.asarray(data["motion"], dtype=np.float32),
            float(data["duration"]),
        )
    except Exception:
        return None


def save_analysis_cache(fingerprint, start, duration, words, segments,
                        rms_per_sec, visual_motion_per_sec, media_duration) -> None:
    """Stores the expensive analysis so a re-run with other settings starts in ~0.1 s."""
    path = analysis_cache_path(fingerprint, start, duration)
    if not path:
        return
    payload = {
        "version": CACHE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "duration": float(media_duration),
        "words": [{"word": w["word"],
                   "start": round(float(w["start"]), 3),
                   "end": round(float(w["end"]), 3)} for w in words],
        "segments": [{"start": round(float(s["start"]), 3),
                      "end": round(float(s["end"]), 3),
                      "text": s["text"]} for s in segments],
        "rms": [round(float(v), 5) for v in rms_per_sec],
        "motion": [round(float(v), 3) for v in visual_motion_per_sec],
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def cleanup_old_files(folder: str, max_age_hours: float = UPLOAD_MAX_AGE_HOURS) -> int:
    """Deletes files older than max_age_hours from folder, plus any empty subfolders."""
    if not folder or not os.path.isdir(folder):
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass
        for name in dirs:
            path = os.path.join(root, name)
            try:
                if not os.listdir(path):
                    os.rmdir(path)
            except OSError:
                pass
    return removed


# --- LLM scoring (Google Gemini) ------------------------------------------

# OpenAI-compatible providers: one connector, ready-made presets, any model id stays editable
LLM_PROVIDERS = {
    "Groq — llama-3.3-70b (бесплатно)": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "site": "https://groq.com",
    },
    "Cerebras — llama-3.3-70b (бесплатно)": {
        "base_url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama-3.3-70b",
        "key_env": "CEREBRAS_API_KEY",
        "site": "https://cerebras.ai",
    },
    "Google Gemini (OpenAI-совместимый)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
        "site": "https://ai.google.dev",
    },
    "Ollama локальная (без интернета)": {
        "base_url": "http://localhost:11434/v1/chat/completions",
        "model": "qwen3:4b",
        "key_env": "OLLAMA_API_KEY",
        "site": "https://ollama.com",
    },
}
DEFAULT_LLM_PROVIDER = "Groq — llama-3.3-70b (бесплатно)"
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_CANDIDATES = 28
# Local models (Ollama/qwen3) have a tiny default context and answer only the first couple
# of items of a big prompt, so they get small batches and explicit context/output options.
LLM_LOCAL_BATCH_SIZE = 4
LLM_LOCAL_NUM_CTX = 8192
LLM_LOCAL_NUM_PREDICT = 2048
LLM_RETRY_BATCH_SIZE = 2
# Fragments the model never scored are dropped instead of being rendered by math score
LLM_DROP_UNEVALUATED = True
# Instant ad guard that needs no model at all: text + ±15 seconds around the fragment
LLM_AD_STOP_PHRASES = (
    "по ссылке в описании",
    "ссылка в описании",
    "ссылку в описании",
    "промокод",
    "рекламной интеграции",
    "рекламная интеграция",
    "спонсор выпуска",
    "партнёр выпуска",
    "партнер выпуска",
    "erid",
)
LLM_AD_SCAN_SECONDS = 15.0
# how many seconds of speech to show the model before and after the fragment
LLM_CONTEXT_SECONDS = 20.0
# one request per run; split only when the prompt would blow past a free-tier token limit
LLM_MAX_PROMPT_CHARS = 24000
LLM_CACHE_SUBDIR = os.path.join(CACHE_DIR, "llm")

LLM_SYSTEM_PROMPT = """Ты — редактор вирусных клипов для TikTok, Reels и Shorts и модератор рекламы.
Тебе дают фрагменты стрима с контекстом до и после.

ПРАВИЛО №1: сначала проверь рекламу, она отсеивается жёстко.
is_ad = true, если в самом фрагменте, на его границе или в контексте есть:
- подводка к интеграции: «сразу после небольшой рекламной интеграции», «перед началом ролика»,
  «спонсор выпуска», «при поддержке», «наш партнёр», «а теперь про...», «не забудьте про...»;
- сама интеграция: промокод, скидка, ссылка в описании, «переходи по ссылке», «заказывайте»,
  бренд, товар или сервис с характеристиками, ценой или преимуществами;
- заставка канала, бэмпер, склейка, перебивка на другую тему, пауза между блоками;
- призывы: подпишись, поставь лайк, донат, розыгрыш, «ссылка в шапке».
Если фрагмент НАЧИНАЕТСЯ сразу после фразы-подводки к интеграции или ОБРЫВАЕТСЯ в самом
начале рекламы — is_ad = true, даже если внутри нет слова «реклама».
Контекст [before]/[after] дан именно для этого: маркер может быть вне самого фрагмента.

score (0-100) — вирусность, только если is_ad = false:
- сильный хук в первые 3 секунды;
- законченная мысль, понятная без контекста;
- эмоция, юмор, конфликт, неожиданный инсайт или конкретная цифра;
- скучный пересказ, вода, обрыв мысли — низкий балл.
Если is_ad = true — score = 0.

title — цепляющий заголовок на русском, 3-6 слов, без кавычек и хэштегов.

Отвечай только JSON, без markdown и без пояснений."""


def load_dotenv(path: str) -> int:
    """Loads KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    loaded = 0
    try:
        if not os.path.isfile(path):
            return 0
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                if name.startswith("export "):
                    name = name[7:].strip()
                value = value.strip().strip('"').strip("'")
                if name and name not in os.environ:
                    os.environ[name] = value
                    loaded += 1
    except Exception:
        return loaded
    return loaded


ENV_FILE = os.path.join(CURRENT_DIR, ".env")
load_dotenv(ENV_FILE)


def llm_provider_preset(provider_name):
    return LLM_PROVIDERS.get(provider_name) or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]


def llm_api_key(provider_name, explicit_key: str = "") -> str:
    """Explicit UI key wins, then the provider's own env var / .env value."""
    key = (explicit_key or "").strip()
    if key:
        return key
    return (os.environ.get(llm_provider_preset(provider_name)["key_env"]) or "").strip()


def llm_provider_fields(provider_name):
    """Values used to prefill model / url / key when the provider dropdown changes."""
    preset = llm_provider_preset(provider_name)
    return preset["model"], preset["base_url"], (os.environ.get(preset["key_env"]) or "")


def _is_local_endpoint(url: str) -> bool:
    lowered = (url or "").lower()
    return "localhost" in lowered or "127.0.0.1" in lowered or "0.0.0.0" in lowered


def _format_mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    return "%02d:%02d" % (total // 60, total % 60)


def _llm_cache_path(model: str, text: str) -> str:
    digest = hashlib.sha1(f"v3|{model}|{text}".encode("utf-8")).hexdigest()
    return os.path.join(LLM_CACHE_SUBDIR, digest + ".json")


def load_llm_verdict(model: str, text: str):
    path = _llm_cache_path(model, text)
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return {
                "score": max(0, min(100, int(data["score"]))),
                "is_ad": bool(data["is_ad"]),
                "title": str(data["title"]),
            }
    except Exception:
        return None
    return None


def save_llm_verdict(model: str, text: str, verdict: dict) -> None:
    try:
        os.makedirs(LLM_CACHE_SUBDIR, exist_ok=True)
        with open(_llm_cache_path(model, text), "w", encoding="utf-8") as handle:
            json.dump(verdict, handle, ensure_ascii=False)
    except Exception:
        pass


def _parse_llm_json(text: str):
    """Parses a JSON answer, cutting off ``` fences, <think> blocks and chatty text."""
    if not text:
        return None
    # qwen3 may emit a reasoning block that eats the whole answer budget
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _call_ollama_native(base_url: str, model: str, system_prompt: str, user_prompt: str,
                        timeout: int = LLM_TIMEOUT_SECONDS):
    """Native Ollama /api/chat: the only way to really pass num_ctx and disable thinking.

    The OpenAI-compatible layer ignores `options`, so qwen3 kept truncating at its default
    2048-token context. Returns parsed JSON or None (then the caller tries the compat API).
    """
    try:
        parts = urllib.parse.urlsplit(base_url)
        if not parts.scheme or not parts.netloc:
            return None
        native_url = "%s://%s/api/chat" % (parts.scheme, parts.netloc)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "num_ctx": LLM_LOCAL_NUM_CTX,
                "num_predict": LLM_LOCAL_NUM_PREDICT,
                "temperature": 0.2,
            },
        }
        request = urllib.request.Request(
            native_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = (payload.get("message") or {}).get("content") or ""
        return _parse_llm_json(content)
    except Exception:
        return None


def call_llm_json(base_url: str, api_key: str, model: str, user_prompt: str,
                  timeout: int = LLM_TIMEOUT_SECONDS, error_sink=None):
    """One chat completion, returns the parsed JSON or None.

    Groq, Cerebras and Gemini speak OpenAI-compatible HTTP; a local Ollama server is asked
    through its native /api/chat first (num_ctx + think=false), then through the compat API.
    Error reasons go to error_sink.
    """

    def _note(message):
        if error_sink is not None:
            error_sink.append(message)

    if not base_url:
        _note("не задан base_url провайдера")
        return None

    local = _is_local_endpoint(base_url)
    system_prompt = LLM_SYSTEM_PROMPT + ("\n\n/no_think" if local else "")

    if local:
        native = _call_ollama_native(base_url, model, system_prompt, user_prompt, timeout)
        if native is not None:
            return native

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if local:
        body["options"] = {"num_ctx": LLM_LOCAL_NUM_CTX, "num_predict": LLM_LOCAL_NUM_PREDICT}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    for attempt in range(3):
        try:
            request = urllib.request.Request(
                base_url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            try:
                content = payload["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                _note("неожиданный формат ответа API")
                return None
            parsed = _parse_llm_json(content)
            if parsed is None:
                _note("модель вернула не JSON")
            return parsed
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", "replace")
            except Exception:
                pass
            short_detail = " ".join(detail.split())[:220]
            # some providers reject response_format — retry once without it
            if (error.code == 400 and "response_format" in body
                    and "response_format" in short_detail.lower()):
                body.pop("response_format", None)
                continue
            if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            _note("HTTP %d: %s" % (error.code, short_detail or "без описания"))
            return None
        except Exception as error:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            _note("%s: %s" % (type(error).__name__, " ".join(str(error).split())[:200]))
            return None
    return None


def _build_llm_prompt(items) -> str:
    """User part of the prompt: every candidate with [before] / [target] / [after]."""
    lines = [
        "Фрагменты стрима. У каждого: время, [before] — контекст до (±20 секунд), "
        "[target] — сам фрагмент, [after] — контекст после.",
        "Сначала проверь рекламу по правилам из системной инструкции, потом оцени вирусность.",
        "",
    ]
    for item in items:
        lines.append("[%d] %s (%.0f c)" % (item["id"], item["timecode"], item["duration"]))
        lines.append("[before] " + (item.get("context_before") or "—"))
        lines.append("[target] " + item["text"])
        lines.append("[after] " + (item.get("context_after") or "—"))
        lines.append("")
    lines.append('Ответь только JSON: {"items":[{"id":1,"score":0,"is_ad":false,"title":""}]}')
    return "\n".join(lines)


def _llm_cache_text(candidate) -> str:
    """Cache identity of a verdict: the fragment plus its surrounding context."""
    return "\n".join([
        " ".join((candidate.get("context_before") or "").split()),
        "||",
        " ".join((candidate.get("text") or "").split()),
        "||",
        " ".join((candidate.get("context_after") or "").split()),
    ])


def _chunk_candidates(pending, max_chars=LLM_MAX_PROMPT_CHARS):
    """One request per run; split only if the prompt would be unreasonably large."""
    chunks = []
    current = []
    size = 0
    for entry in pending:
        cost = len(entry[1]) + 80
        if current and size + cost > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(entry)
        size += cost
    if current:
        chunks.append(current)
    return chunks


def _ad_marker(text: str) -> str:
    """Returns the matched ad stop phrase, or an empty string."""
    lowered = (text or "").lower()
    for phrase in LLM_AD_STOP_PHRASES:
        if phrase in lowered:
            return phrase
    return ""


def apply_llm_scores(settings: dict, candidates) -> dict:
    """Hard ad pre-filter + cache + LLM verdicts, with retries for half-answered batches.

    settings: {base_url, api_key, model, label, local}. Local models are queried in small
    batches and every unanswered candidate is retried in pairs, so a short answer never
    leaves a fragment unverified.
    """
    model = (settings.get("model") or "").strip()
    base_url = (settings.get("base_url") or "").strip()
    api_key = (settings.get("api_key") or "").strip()
    local = bool(settings.get("local")) or _is_local_endpoint(base_url)
    summary = {"ok": False, "evaluated": 0, "from_cache": 0, "requested": 0, "ads": 0,
               "prefiltered": 0, "skipped": 0, "error": "", "label": settings.get("label", "")}
    if not base_url or not model:
        summary["error"] = "не задан base_url или модель"
        return summary
    if not candidates:
        summary["error"] = "нет кандидатов для оценки"
        return summary

    pending = []
    for index, candidate in enumerate(candidates):
        if not (candidate.get("text") or "").strip():
            continue
        # instant guard: promo links and ad lead-ins are caught without asking any model
        scan_text = candidate.get("ad_scan_text") or " ".join([
            candidate.get("context_before") or "",
            candidate.get("text") or "",
            candidate.get("context_after") or "",
        ])
        marker = _ad_marker(scan_text)
        if marker:
            candidate["llm"] = {"score": 0, "is_ad": True, "title": "",
                                "reason": "стоп-фраза: " + marker}
            summary["prefiltered"] += 1
            continue
        cache_text = _llm_cache_text(candidate)
        verdict = load_llm_verdict(model, cache_text)
        if verdict is not None:
            candidate["llm"] = verdict
            summary["from_cache"] += 1
        else:
            pending.append((index, cache_text))

    error_sink = []

    def evaluate(entries, size=None):
        """Sends entries in batches and returns those the model did not answer."""
        unanswered = []
        if size:
            groups = [entries[i:i + size] for i in range(0, len(entries), size)]
        else:
            groups = _chunk_candidates(entries)
        for group in groups:
            items = []
            for local_id, (index, _cache_text) in enumerate(group, start=1):
                candidate = candidates[index]
                items.append({
                    "id": local_id,
                    "timecode": "%s–%s" % (_format_mmss(candidate["start"]), _format_mmss(candidate["end"])),
                    "duration": candidate["duration"],
                    "text": " ".join((candidate.get("text") or "").split()),
                    "context_before": " ".join((candidate.get("context_before") or "").split()),
                    "context_after": " ".join((candidate.get("context_after") or "").split()),
                })
            summary["requested"] += len(group)
            result = call_llm_json(base_url, api_key, model, _build_llm_prompt(items),
                                   error_sink=error_sink)
            if not result:
                unanswered.extend(group)
                continue
            by_id = {}
            for entry in result.get("items") or []:
                try:
                    by_id[int(entry["id"])] = entry
                except (KeyError, TypeError, ValueError):
                    continue
            for local_id, (index, cache_text) in enumerate(group, start=1):
                entry = by_id.get(local_id)
                if entry is None:
                    unanswered.append((index, cache_text))
                    continue
                try:
                    score = max(0, min(100, int(round(float(entry.get("score", 0))))))
                except (TypeError, ValueError):
                    score = 0
                is_ad = bool(entry.get("is_ad", False))
                verdict = {
                    "score": 0 if is_ad else score,
                    "is_ad": is_ad,
                    "title": str(entry.get("title", "")).strip().strip('"'),
                }
                candidates[index]["llm"] = verdict
                save_llm_verdict(model, cache_text, verdict)
                summary["evaluated"] += 1
        return unanswered

    remaining = pending
    if remaining:
        remaining = evaluate(remaining, LLM_LOCAL_BATCH_SIZE if local else None)
    if remaining:
        # the model answered only part of a batch: retry the rest two at a time
        remaining = evaluate(remaining, LLM_RETRY_BATCH_SIZE)
    if remaining:
        remaining = evaluate(remaining, 1)

    summary["skipped"] = len(remaining)
    summary["ads"] = sum(1 for c in candidates if (c.get("llm") or {}).get("is_ad"))
    summary["ok"] = (summary["evaluated"] + summary["from_cache"] + summary["prefiltered"]) > 0
    if not summary["ok"]:
        summary["error"] = error_sink[-1] if error_sink else "модель не вернула ответ"
    return summary


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def _ascii_slug(text: str, limit: int = 48) -> str:
    """Latin filename-safe version of a Russian title (URLs and Windows stay happy)."""
    if not text:
        return ""
    transliterated = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    return re.sub(r"[^a-z0-9]+", "_", transliterated).strip("_")[:limit]


def get_file_path(file_input) -> str:
    """Extracts a valid file path string from various Gradio input formats (dict, list, object, str)."""
    if file_input is None:
        return ""
    if isinstance(file_input, (list, tuple)):
        for item in file_input:
            path = get_file_path(item)
            if path:
                return path
        return ""
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict):
        if "path" in file_input and file_input["path"]:
            return file_input["path"]
        if "name" in file_input and file_input["name"]:
            return file_input["name"]
        if "video" in file_input and isinstance(file_input["video"], dict):
            v_dict = file_input["video"]
            return v_dict.get("path") or v_dict.get("name") or ""
    if hasattr(file_input, "name") and file_input.name:
        return str(file_input.name)
    if hasattr(file_input, "path") and file_input.path:
        return str(file_input.path)
    return str(file_input)


UPLOAD_HINT = "⚠️ Загрузите видео или укажите полный путь к файлу на диске."


def resolve_video_source(video_input, local_path_input):
    """Chooses the video to process: an explicit disk path wins over the uploaded file.

    Typing a path keeps multi-GB sources away from the browser upload and from the
    Gradio temp folder completely. Returns (path, error_message).
    """
    typed = str(local_path_input or "").strip().strip('"').strip("'")
    if typed:
        typed_path = os.path.abspath(os.path.expanduser(typed))
        if os.path.isfile(typed_path):
            return typed_path, ""
        fallback = get_file_path(video_input)
        if fallback and os.path.isfile(fallback):
            return fallback, ""
        return "", f"⚠️ Файл по пути «{typed}» не найден. Проверьте путь или загрузите видео."

    uploaded = get_file_path(video_input)
    if uploaded and os.path.isfile(uploaded):
        return uploaded, ""
    return "", UPLOAD_HINT


def isolate_and_copy_video(src_path: str, work_dir: str) -> str:
    """Copies src_path into work_dir under an ASCII name (source_<uuid>.<ext>), retrying
    while Windows still holds the freshly uploaded file open, so FFmpeg never has to
    read a locked Cyrillic-named file straight out of the Gradio temp folder.
    """
    if not os.path.isfile(src_path):
        return src_path

    ext = os.path.splitext(src_path)[1] or ".mp4"
    dest_path = os.path.join(work_dir, f"source_{uuid.uuid4().hex}{ext}")

    for attempt in range(5):
        cmd = [FFMPEG_BIN, "-y", "-i", src_path, "-c", "copy", dest_path]
        try:
            run_subprocess_cancellable(cmd)
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
                return dest_path
        except Exception:
            pass
        # the upload may still be locked by the browser / antivirus — wait a moment
        time.sleep(0.6)
        dest_path = os.path.join(work_dir, f"source_{uuid.uuid4().hex}{ext}")

    last_error = None
    for _ in range(5):
        try:
            with open(src_path, "rb") as f_src:
                with open(dest_path, "wb") as f_dst:
                    shutil.copyfileobj(f_src, f_dst, 1024 * 1024)
            if os.path.getsize(dest_path) > 0:
                return dest_path
        except OSError as exc:
            last_error = exc
            time.sleep(0.6)

    raise RuntimeError(
        f"Не удалось прочитать файл «{os.path.basename(src_path)}»: он занят другой программой "
        f"({last_error}). Укажите путь к файлу вручную или закройте программу, которая его держит."
    )


def parse_time_value(time_str):
    """Parses a time value ('45.5', '1:25', '01:25:30') into seconds, or None when unparsable.

    An empty field is valid and means 0.0 — that is how "no manual range" is expressed.
    """
    if time_str is None:
        return 0.0
    if isinstance(time_str, (int, float)):
        return float(time_str) if time_str >= 0 else None

    s = str(time_str).strip().replace(",", ".")
    if not s:
        return 0.0

    try:
        value = float(s)
    except ValueError:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                value = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                value = float(parts[0]) * 60 + float(parts[1])
            else:
                return None
        except ValueError:
            return None

    return value if value >= 0 else None


def parse_time_to_seconds(time_str) -> float:
    """Parses time string (e.g. '1:25', '01:25:30', '45.5') into total seconds, 0.0 if invalid."""
    value = parse_time_value(time_str)
    return 0.0 if value is None else value


def escape_ffmpeg_path(path: str) -> str:
    """Escapes file paths properly for FFmpeg filter arguments on Windows."""
    norm_path = path.replace("\\", "/")
    escaped = norm_path.replace(":", "\\:").replace("'", "\\'")
    return escaped


def extract_audio(video_path: str, output_wav: str, start_time: float = 0.0, duration: float = None) -> None:
    """Extracts 16kHz mono PCM WAV audio from video using FFmpeg, optionally within time range."""
    cmd = [FFMPEG_BIN, "-y"]
    if start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    
    cmd.extend(["-i", video_path])
    
    if duration is not None and duration > 0:
        cmd.extend(["-t", f"{duration:.3f}"])
        
    cmd.extend([
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav
    ])
    run_subprocess_cancellable(cmd)


def compute_rms_per_second(wav_path: str) -> np.ndarray:
    """Computes RMS volume per 1-second frame using SciPy and NumPy."""
    sample_rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    total_samples = len(data)
    total_seconds = int(np.ceil(total_samples / sample_rate))
    if total_seconds == 0:
        return np.array([0.0], dtype=np.float32)

    rms_per_sec = []
    for sec in range(total_seconds):
        start = sec * sample_rate
        end = min((sec + 1) * sample_rate, total_samples)
        chunk = data[start:end]
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk ** 2))
        else:
            rms = 0.0
        rms_per_sec.append(float(rms))

    return np.array(rms_per_sec, dtype=np.float32)


def _spread_diffs_over_seconds(diffs, timestamps, duration) -> np.ndarray:
    """Puts keyframe-to-keyframe diffs on a 1 Hz grid using their real timestamps.

    timestamps[i] is the position of frame i in seconds from the analysed range start,
    diffs[i] measures motion between frame i and frame i+1.
    """
    count = len(diffs)
    if count == 0:
        return np.array([], dtype=np.float32)

    if len(timestamps) == count + 1:
        marks = timestamps[1:]
    else:
        span = float(duration) if duration and duration > 0 else float(count)
        marks = [(i + 1) * span / float(count) for i in range(count)]

    total_seconds = max(1, int(math.ceil(max(marks[-1], float(duration or 0.0)))))
    grid = np.full(total_seconds, np.nan, dtype=np.float32)
    for mark, diff in zip(marks, diffs):
        index = min(total_seconds - 1, max(0, int(mark)))
        grid[index] = float(diff)

    # forward-fill, then fill the leading gap with the first measured value
    last_value = np.nan
    for i in range(total_seconds):
        if np.isnan(grid[i]):
            grid[i] = last_value
        else:
            last_value = grid[i]

    first_value = 0.0
    for value in grid:
        if not np.isnan(value):
            first_value = value
            break
    grid[np.isnan(grid)] = first_value
    return grid


def compute_visual_motion_per_second(video_path: str, start_time: float = 0.0, duration: float = None) -> np.ndarray:
    """Computes a per-second visual motion score from low-res grayscale frame diffs.

    With MOTION_KEYFRAMES_ONLY only keyframes are decoded (measured ~9x faster on this
    1080p60 source); showinfo timestamps then place every diff on the 1 Hz grid, so the
    scorer keeps seeing exactly one motion value per second.
    """
    cmd = [FFMPEG_BIN, "-y"]
    if MOTION_KEYFRAMES_ONLY:
        cmd.extend(["-skip_frame", "nokey"])
    if start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    cmd.extend(["-i", video_path])
    if duration is not None and duration > 0:
        cmd.extend(["-t", f"{duration:.3f}"])

    if MOTION_KEYFRAMES_ONLY:
        # fps_mode passthrough: otherwise rawvideo pads the decoded keyframes back to 60 fps
        cmd.extend(["-vf", "scale=160:90,showinfo", "-fps_mode", "passthrough"])
    else:
        cmd.extend(["-vf", "scale=160:90,fps=1"])

    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ])

    try:
        proc = _popen_ffmpeg(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             log_level="info" if MOTION_KEYFRAMES_ONLY else "error")
        err_chunks = []
        drainer = threading.Thread(target=_drain_pipe, args=(proc.stderr, err_chunks), daemon=True)
        drainer.start()

        raw_data = _read_stdout_cancellable(proc, 160 * 90 * 10)
        if STOP_EVENT.is_set():
            _terminate_proc(proc)
            return np.array([], dtype=np.float32)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_proc(proc)
        drainer.join(timeout=2)
        if proc.returncode != 0:
            return np.array([], dtype=np.float32)

        frame_size = 160 * 90
        num_frames = len(raw_data) // frame_size
        if num_frames < 2:
            return np.array([50.0], dtype=np.float32)

        frames = np.frombuffer(raw_data[:num_frames * frame_size], dtype=np.uint8).reshape((num_frames, frame_size))
        diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean(axis=1)

        if MOTION_KEYFRAMES_ONLY:
            stderr_text = b"".join(err_chunks).decode("utf-8", "replace")
            timestamps = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr_text)]
            motion_per_sec = _spread_diffs_over_seconds(diffs, timestamps, duration)
        else:
            motion_per_sec = np.concatenate(([diffs[0]], diffs))

        if len(motion_per_sec) == 0:
            return np.array([50.0], dtype=np.float32)

        max_diff = float(np.max(motion_per_sec))
        if max_diff > 0:
            return (motion_per_sec / max_diff) * 100.0
        return np.full(len(motion_per_sec), 50.0, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def face_center_track(video_path, start_t, end_t):
    """Follows the largest face across the clip: [(t_rel, x_norm), ...] at FACE_TRACK_FPS.

    A single ffmpeg pass at FACE_TRACK_WIDTH×FACE_TRACK_HEIGHT is enough for a crop that
    only pans horizontally, and keeps YuNet fast (~15 ms per frame).
    """
    if not face_detection_available():
        return []
    clip_dur = end_t - start_t
    if clip_dur <= 0:
        return []

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start_t:.3f}",
        "-to", f"{end_t:.3f}",
        "-i", video_path,
        "-vf", f"scale={FACE_TRACK_WIDTH}:{FACE_TRACK_HEIGHT},fps={FACE_TRACK_FPS}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    frame_size = FACE_TRACK_WIDTH * FACE_TRACK_HEIGHT * 3
    samples = []
    try:
        proc = _popen_ffmpeg(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=_drain_pipe, args=(proc.stderr, []), daemon=True).start()

        raw = _read_stdout_cancellable(proc, frame_size * 4)
        if STOP_EVENT.is_set():
            _terminate_proc(proc)
            return []
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_proc(proc)
        if proc.returncode != 0:
            return []

        num_frames = len(raw) // frame_size
        detector = _get_face_detector(FACE_TRACK_WIDTH, FACE_TRACK_HEIGHT)
        if detector is None or num_frames == 0:
            return []

        for index in range(num_frames):
            if STOP_EVENT.is_set():
                break
            frame = np.frombuffer(raw[index * frame_size:(index + 1) * frame_size],
                                  dtype=np.uint8).reshape((FACE_TRACK_HEIGHT, FACE_TRACK_WIDTH, 3))
            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue
            # a corner webcam overlay is a few percent wide — do not chase it on gameplay
            qualifying = [f for f in faces if f[2] >= FACE_TRACK_WIDTH * FACE_TRACK_MIN_RATIO]
            if not qualifying:
                continue
            total_area = sum(f[2] * f[3] for f in qualifying) or 1.0
            centre_x = sum((f[0] + f[2] / 2.0) * (f[2] * f[3]) for f in qualifying) / total_area
            samples.append((index / float(FACE_TRACK_FPS),
                            float(min(1.0, max(0.0, centre_x / float(FACE_TRACK_WIDTH))))))
    except Exception:
        return samples
    return samples


def compute_horizontal_tracking_expression(video_path: str, start_t: float, end_t: float) -> str:
    """Builds a smooth FFmpeg x-expression that keeps the streamer's face in frame.

    Positions come from the local YuNet detector, then a dead-zone + exponential smoothing
    keeps the camera from twitching on head micro-movements. No face at all → centred crop.
    """
    clip_dur = end_t - start_t
    if clip_dur <= 0:
        return "(iw-ow)/2"

    try:
        samples = face_center_track(video_path, start_t, end_t)
        if not samples:
            return "(iw-ow)/2"

        smoothed = []
        current = samples[0][1]
        for _time, value in samples:
            if abs(value - current) > FACE_DEAD_ZONE:
                current = current * (1.0 - FACE_SMOOTHING) + value * FACE_SMOOTHING
            smoothed.append(current)

        points = [(sample[0], round(value, 4)) for sample, value in zip(samples, smoothed)]
        if not points:
            return "(iw-ow)/2"

        def build_nested_expr(pts, idx):
            if idx >= len(pts) - 1:
                return f"{pts[-1][1]:.4f}"
            t0, r0 = pts[idx]
            t1, r1 = pts[idx + 1]
            dt = max(0.001, t1 - t0)
            dr = r1 - r0
            interp = f"({r0:.4f}+{dr:.4f}*(t-{t0:.2f})/{dt:.2f})"
            if idx == len(pts) - 2:
                return f"if(lte(t,{t1:.2f}),{interp},{r1:.4f})"
            return f"if(lte(t,{t1:.2f}),{interp},{build_nested_expr(pts, idx + 1)})"

        rel_x_expr = build_nested_expr(points, 0)
        # crop so the face sits in the middle: x = rel*iw - ow/2, clamped to the frame
        return f"clip({rel_x_expr}*iw-ow/2,0,iw-ow)"

    except Exception:
        return "(iw-ow)/2"


_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()


def get_whisper_model():
    """Loads the Whisper model once per process instead of rebuilding it on every run."""
    global _WHISPER_MODEL
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            _WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def transcribe_audio(wav_path: str):
    """Transcribes audio using faster-whisper with word timestamps."""
    if STOP_EVENT.is_set():
        return [], [], 0.0
    model = get_whisper_model()
    segments_generator, info = model.transcribe(wav_path, word_timestamps=True, language="ru")
    
    words = []
    segments = []
    for segment in segments_generator:
        if STOP_EVENT.is_set():
            break
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        if segment.words:
            for word_item in segment.words:
                words.append({
                    "word": word_item.word.strip(),
                    "start": word_item.start,
                    "end": word_item.end
                })

    return words, segments, info.duration


def score_clip_candidate(words, rms_per_sec, visual_motion_per_sec, start_t, end_t, total_duration, time_offset=0.0):
    """Scores a candidate segment from 0 to 100 based on hook, speech density, RMS volume, and visual motion.

    start_t / end_t live on the absolute source timeline, while rms_per_sec and
    visual_motion_per_sec are indexed from the beginning of the analysed range.
    time_offset (the range start, i.e. the manual "-ss" value) converts one into the other;
    without it volume and motion scores silently collapse to the 50.0 default.
    """
    clip_len = end_t - start_t
    if clip_len <= 0:
        return 0.0

    clip_words = [w for w in words if start_t <= w["start"] <= end_t]
    
    hook_words = [w for w in clip_words if w["start"] <= start_t + 5.0]
    hook_text = " ".join([w["word"].lower() for w in hook_words])
    
    hook_found_count = sum(1 for hw in HOOK_WORDS if hw in hook_text)
    has_question_or_excl = "?" in hook_text or "!" in hook_text
    
    hook_score = min(100.0, hook_found_count * 30.0 + (30.0 if has_question_or_excl else 0.0) + (len(hook_words) * 4.0))

    words_per_sec = len(clip_words) / clip_len
    density_score = min(100.0, (words_per_sec / 3.2) * 85.0)

    s_sec = max(0, int(math.floor(start_t - time_offset)))
    e_sec = max(0, int(math.ceil(end_t - time_offset)))
    if len(rms_per_sec) > 0 and e_sec > s_sec:
        clip_rms = rms_per_sec[s_sec:min(e_sec, len(rms_per_sec))]
        if len(clip_rms) > 0:
            avg_rms = float(np.mean(clip_rms))
            max_rms = float(np.max(clip_rms))
            global_avg = float(np.mean(rms_per_sec)) if np.mean(rms_per_sec) > 0 else 0.01
            rms_ratio = (avg_rms / global_avg) if global_avg > 0 else 1.0
            peak_ratio = (max_rms / global_avg) if global_avg > 0 else 1.0
            rms_score = min(100.0, (rms_ratio * 50.0) + (peak_ratio * 30.0))
        else:
            rms_score = 50.0
    else:
        rms_score = 50.0

    if len(visual_motion_per_sec) > 0 and e_sec > s_sec:
        clip_motion = visual_motion_per_sec[s_sec:min(e_sec, len(visual_motion_per_sec))]
        if len(clip_motion) > 0:
            avg_motion = float(np.mean(clip_motion))
            global_avg_motion = float(np.mean(visual_motion_per_sec)) if np.mean(visual_motion_per_sec) > 0 else 1.0
            motion_ratio = (avg_motion / global_avg_motion)
            motion_score = min(100.0, motion_ratio * 70.0)
        else:
            motion_score = 50.0
    else:
        motion_score = 50.0

    composite = (0.35 * hook_score) + (0.30 * density_score) + (0.20 * rms_score) + (0.15 * motion_score)
    final_score = int(round(min(100.0, max(0.0, composite))))
    return final_score


def find_juicy_clips(words, segments, rms_per_sec, visual_motion_per_sec, total_duration, juicy_threshold=75, target_len=40, max_clips=0, range_min=0.0, range_max=None, time_offset=0.0):
    """Scans the video, generates pause-aligned candidates, scores them, adds neat padding, and filters non-overlapping moments."""
    if not words or total_duration < 10:
        return []

    try:
        max_clips = int(max_clips) if max_clips is not None else 0
    except (TypeError, ValueError):
        max_clips = 0
    if max_clips < 0:
        max_clips = 0

    if range_max is None:
        range_max = total_duration

    pauses = [0.0]
    for i in range(len(words) - 1):
        gap = words[i+1]["start"] - words[i]["end"]
        if gap >= 0.25:
            pauses.append(words[i]["end"])
    pauses.append(total_duration)

    candidates = []
    min_len = max(15, target_len - 10)
    max_len = min(60, target_len + 10)

    for start_p in pauses:
        if start_p >= total_duration - min_len:
            break
        for end_p in pauses:
            if end_p <= start_p:
                continue
            duration = end_p - start_p
            if min_len <= duration <= max_len:
                score = score_clip_candidate(words, rms_per_sec, visual_motion_per_sec, start_p, end_p, total_duration, time_offset=time_offset)
                if score >= juicy_threshold:
                    candidates.append({
                        "start": start_p,
                        "end": end_p,
                        "duration": duration,
                        "score": score
                    })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    PADDING_SEC = 0.5

    for cand in candidates:
        overlap = False
        for sel in selected:
            if not (cand["end"] <= sel["start"] or cand["start"] >= sel["end"]):
                overlap = True
                break
        if not overlap:
            padded_start = max(range_min, cand["start"] - PADDING_SEC)
            padded_end = min(range_max, cand["end"] + CLIP_TAIL_SECONDS)
            
            cand["start"] = padded_start
            cand["end"] = padded_end
            cand["duration"] = padded_end - padded_start
            selected.append(cand)

    # Keep the N best moments by score first, then show them in chronological order.
    selected.sort(key=lambda x: x["score"], reverse=True)
    if max_clips > 0:
        selected = selected[:max_clips]
    selected.sort(key=lambda x: x["start"])

    return selected


def _collect_context(segments, position, step, edge, limit):
    """Text of up to `limit` seconds of segments adjacent to one candidate edge."""
    parts = []
    while 0 <= position < len(segments):
        segment = segments[position]
        if step < 0:
            if edge - segment["start"] > limit:
                break
        else:
            if segment["end"] - edge > limit:
                break
        text = (segment.get("text") or "").strip()
        if text:
            parts.append(text)
        position += step
    if step < 0:
        parts.reverse()
    return " ".join(" ".join(parts).split())


def find_sentence_clips(words, segments, target_len=40, range_min=0.0, range_max=None, time_offset=0.0,
                        context_seconds=LLM_CONTEXT_SECONDS):
    """Sentence-aligned candidates: whole Whisper segments, never a cut in the middle of a phrase.

    For every segment start it keeps the window whose length is closest to target_len and
    still inside [target_len - 10, target_len + 15]. Each candidate also carries the speech
    right before and after it, so the LLM can spot an ad lead-in or a lead-out outside the cut.
    """
    if not segments:
        return []
    if range_max is None:
        range_max = segments[-1]["end"]

    min_len = max(12.0, float(target_len) - 10.0)
    max_len = min(90.0, float(target_len) + 15.0)

    usable = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        if segment["end"] <= range_min + 0.5 or segment["start"] >= range_max - 0.5:
            continue
        usable.append(segment)

    candidates = []
    count = len(usable)
    for i in range(count):
        start = max(range_min, usable[i]["start"])
        best_delta = None
        best = None
        best_end = i
        for j in range(i, count):
            end = min(range_max, usable[j]["end"])
            duration = end - start
            if duration > max_len:
                break
            if duration < min_len:
                continue
            delta = abs(duration - float(target_len))
            if best_delta is None or delta < best_delta:
                best_delta = delta
                text = " ".join((usable[k].get("text") or "").strip() for k in range(i, j + 1))
                best = (start, end, " ".join(text.split()))
                best_end = j
        if best is not None:
            clip_end = min(range_max, best[1] + CLIP_TAIL_SECONDS)
            candidates.append({
                "start": best[0],
                "end": clip_end,
                "duration": clip_end - best[0],
                "score": 0,
                "text": best[2],
                "context_before": _collect_context(usable, i - 1, -1, best[0], context_seconds),
                "context_after": _collect_context(usable, best_end + 1, 1, best[1], context_seconds),
                # narrower ±15 s window used by the instant ad stop-phrase filter
                "ad_scan_text": " ".join([
                    _collect_context(usable, i - 1, -1, best[0], LLM_AD_SCAN_SECONDS),
                    best[2],
                    _collect_context(usable, best_end + 1, 1, best[1], LLM_AD_SCAN_SECONDS),
                ]),
            })
    return candidates


def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds to ASS timestamp H:MM:SS.cs"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int(round((seconds - int(seconds)) * 100))
    if centisecs == 100:
        centisecs = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{centisecs:02d}"


WEBCAM_BASE_FRACTION = 0.35
# "По центру слева/справа": the search zone is a third of the width, vertically 20-80%
WEBCAM_SIDE_FRACTION = 1.0 / 3.0
WEBCAM_CENTRE_TOP = 0.20
WEBCAM_CENTRE_BOTTOM = 0.80

# --- Face-based reframe of the main crop ---
FACE_TRACK_FPS = 2
FACE_TRACK_WIDTH = 640
FACE_TRACK_HEIGHT = 360
FACE_DEAD_ZONE = 0.04        # ignore head micro-moves (fraction of the frame width)
FACE_SMOOTHING = 0.25        # exponential smoothing factor for the camera path
# only faces this large are treated as the subject: tiny ones are webcam overlays on gameplay
FACE_TRACK_MIN_RATIO = 0.08

# breathing room after the last word so the final phrase is not cut mid-syllable
CLIP_TAIL_SECONDS = 0.6

# single-pass fallback when measurement fails; the accurate path uses measured_* values
AUDIO_LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"

# analysis / LLM caches older than this are removed at startup
CACHE_MAX_AGE_DAYS = 30
FACE_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "face_detection_yunet_2023mar.onnx")

_MEDIA_SIZE_CACHE = {}
_HAS_AUDIO_CACHE = {}
_CV2_MODULE = None
_CV2_CHECKED = False
_FACE_DETECTOR = None
_FACE_CACHE = {}


def probe_video_size(path: str):
    """Returns (width, height) of the first video stream, cached per path."""
    if path in _MEDIA_SIZE_CACHE:
        return _MEDIA_SIZE_CACHE[path]
    size = None
    try:
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = CREATE_NO_WINDOW
        result = subprocess.run([FFMPEG_BIN, "-hide_banner", "-nostdin", "-i", path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60, **kwargs)
        match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", result.stderr.decode("utf-8", "replace"))
        if match:
            size = (int(match.group(1)), int(match.group(2)))
    except Exception:
        size = None
    _MEDIA_SIZE_CACHE[path] = size
    return size


def probe_has_audio(path: str) -> bool:
    """True when the file carries at least one audio stream (cached per path)."""
    if path in _HAS_AUDIO_CACHE:
        return _HAS_AUDIO_CACHE[path]
    has_audio = False
    try:
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = CREATE_NO_WINDOW
        result = subprocess.run([FFMPEG_BIN, "-hide_banner", "-nostdin", "-i", path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60, **kwargs)
        has_audio = "Audio:" in result.stderr.decode("utf-8", "replace")
    except Exception:
        has_audio = False
    _HAS_AUDIO_CACHE[path] = has_audio
    return has_audio


def _load_cv2():
    """Imports OpenCV once; returns None when it is not installed (feature degrades)."""
    global _CV2_MODULE, _CV2_CHECKED
    if not _CV2_CHECKED:
        _CV2_CHECKED = True
        try:
            import cv2  # noqa: PLC0415
            _CV2_MODULE = cv2
        except Exception:
            _CV2_MODULE = None
    return _CV2_MODULE


def face_detection_available() -> bool:
    return _load_cv2() is not None and os.path.isfile(FACE_MODEL_PATH)


def _get_face_detector(width, height):
    global _FACE_DETECTOR
    cv2 = _load_cv2()
    if cv2 is None or not os.path.isfile(FACE_MODEL_PATH):
        return None
    try:
        if _FACE_DETECTOR is None:
            _FACE_DETECTOR = cv2.FaceDetectorYN.create(FACE_MODEL_PATH, "", (int(width), int(height)),
                                                      score_threshold=0.5, nms_threshold=0.3, top_k=20)
        else:
            _FACE_DETECTOR.setInputSize((int(width), int(height)))
        return _FACE_DETECTOR
    except Exception:
        return None


def detect_face_group(video_path, region, sample_time):
    """Returns (centre_x, centre_y) of the biggest detected face inside region, or None.

    Region is (x, y, w, h) in source pixels; the frame is sampled once at sample_time.
    """
    key = (video_path, tuple(int(v) for v in region), round(float(sample_time), 1))
    if key in _FACE_CACHE:
        return _FACE_CACHE[key]

    result = None
    cv2 = _load_cv2()
    try:
        if cv2 is not None and os.path.isfile(FACE_MODEL_PATH):
            capture = cv2.VideoCapture(video_path)
            try:
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(sample_time)) * 1000.0)
                    ok, frame = capture.read()
                else:
                    ok, frame = False, None
            finally:
                capture.release()

            if ok and frame is not None:
                x0, y0, box_w, box_h = (int(v) for v in region)
                x0 = max(0, x0)
                y0 = max(0, y0)
                box_w = min(box_w, frame.shape[1] - x0)
                box_h = min(box_h, frame.shape[0] - y0)
                if box_w > 16 and box_h > 16:
                    detector = _get_face_detector(box_w, box_h)
                    if detector is not None:
                        _, faces = detector.detect(frame[y0:y0 + box_h, x0:x0 + box_w])
                        if faces is not None and len(faces) > 0:
                            main = max(faces, key=lambda f: f[2] * f[3])
                            result = (float(x0 + main[0] + main[2] / 2.0),
                                      float(y0 + main[1] + main[3] / 2.0))
    except Exception:
        result = None

    _FACE_CACHE[key] = result
    return result


def _webcam_layout(position):
    """Splits a position label into (mode, anchored_left, anchored_top).

    mode "corner" means the base box hugs that corner, "side" means the centre-left /
    centre-right zone (left or right third of the frame, vertically centred).
    """
    label = (position or "").strip().lower()
    is_left = "лев" in label
    if "центр" in label:
        return ("side", is_left, True)
    return ("corner", is_left, (position or "").strip().startswith("Верхний"))


def webcam_base_box(src_size, corner, target_w, webcam_h):
    """Allowable area (x, y, w, h) the webcam crop may slide in for this position."""
    src_w, src_h = src_size
    cam_ar = target_w / float(max(1, webcam_h))
    mode, is_left, anchored_top = _webcam_layout(corner)

    if mode == "side":
        zone_w = src_w * WEBCAM_SIDE_FRACTION
        zone_h = src_h * (WEBCAM_CENTRE_BOTTOM - WEBCAM_CENTRE_TOP)
        base_w = min(zone_w, zone_h * cam_ar)
        base_h = base_w / cam_ar
        x0 = 0.0 if is_left else src_w - base_w
        y0 = src_h * WEBCAM_CENTRE_TOP + (zone_h - base_h) / 2.0
    else:
        base_w = min(src_w * WEBCAM_BASE_FRACTION, src_h * WEBCAM_BASE_FRACTION * cam_ar)
        base_h = base_w / cam_ar
        x0 = 0.0 if is_left else src_w - base_w
        y0 = 0.0 if anchored_top else src_h - base_h

    x0 = max(0, min(int(round(x0)), max(0, int(src_w - base_w))))
    y0 = max(0, min(int(round(y0)), max(0, int(src_h - base_h))))
    return (x0, y0, int(base_w), int(base_h))


def find_webcam_face(video_path, src_size, position, target_w, webcam_h, start_t, end_t):
    """Searches the webcam zone for a streamer face at 50/25/75% of the clip."""
    if not src_size:
        return None
    base_box = webcam_base_box(src_size, position, target_w, webcam_h)
    clip_len = max(0.0, end_t - start_t)
    for fraction in (0.5, 0.25, 0.75):
        point = detect_face_group(video_path, base_box, start_t + clip_len * fraction)
        if point is not None:
            return point
    return None


def webcam_crop_box(src_size, corner, target_w, webcam_h, zoom=1.0, offset_x=0.0, offset_y=0.0, face=None):
    """Crop rectangle for the webcam block inside its position zone.

    zoom shrinks the box, offsets slide it from the outer edge toward the centre, and a
    face point (source pixels) overrides the offsets while staying inside the zone.
    Returns even-sized (x, y, w, h) or None when the source size is unknown.
    """
    if not src_size:
        return None
    src_w, src_h = src_size
    cam_ar = target_w / float(max(1, webcam_h))
    region_x, region_y, region_w, region_h = webcam_base_box(src_size, corner, target_w, webcam_h)
    _, is_left, anchored_top = _webcam_layout(corner)

    zoom = max(1.0, float(zoom or 1.0))
    w = max(2, min(int(region_w / zoom) // 2 * 2, src_w // 2 * 2))
    h = max(2, min(int(round(w / cam_ar)) // 2 * 2, src_h // 2 * 2))
    slack_x = max(0, region_w - w)
    slack_y = max(0, region_h - h)

    if face is not None:
        x = int(round(face[0] - w / 2.0))
        y = int(round(face[1] - h / 2.0))
    else:
        offset_x = max(0.0, min(100.0, float(offset_x or 0.0)))
        offset_y = max(0.0, min(100.0, float(offset_y or 0.0)))
        dx = int(round(slack_x * offset_x / 100.0))
        dy = int(round(slack_y * offset_y / 100.0))
        x = region_x + (dx if is_left else slack_x - dx)
        y = region_y + (dy if anchored_top else slack_y - dy)

    # keep the window inside its zone and inside the frame
    x = max(region_x, min(x, region_x + slack_x))
    y = max(region_y, min(y, region_y + slack_y))
    x = max(0, min(x, src_w - w))
    y = max(0, min(y, src_h - h))
    return (x // 2 * 2, y // 2 * 2, w, h)


SUBTITLE_PRESETS = {
    "Hormozi (жёлтый акцент)": {
        "font": "Arial Black",
        "primary": "&H00FFFFFF",
        "accent": "&H0000FFFF",
        "outline_width": 5,
        "shadow": 3,
        "pop": True,
    },
    "Неон (зелёный акцент)": {
        "font": "Arial Black",
        "primary": "&H00FFFFFF",
        "accent": "&H0014FF39",
        "outline_width": 5,
        "shadow": 2,
        "pop": True,
    },
    "Чистый белый с тенью": {
        "font": "Arial",
        "primary": "&H00FFFFFF",
        "accent": "&H00FFFFFF",
        "outline_width": 4,
        "shadow": 4,
        "pop": False,
    },
}

DEFAULT_SUBTITLE_STYLE = "Hormozi (жёлтый акцент)"


def _group_subtitle_phrases(words, max_chars, max_duration=4.0):
    """Splits words into phrases of at most two lines, breaking on punctuation or duration."""
    phrases = []
    current = []
    limit = max_chars * 2
    for word in words:
        if current:
            prospective = " ".join(w["word"] for w in current + [word])
            duration = word["end"] - current[0]["start"]
            if len(prospective) > limit or duration > max_duration:
                phrases.append(current)
                current = []
        current.append(word)
        if word["word"].rstrip().endswith((".", "!", "?", "…", ":", ";")):
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def _wrap_phrase_lines(words, max_chars):
    """Chooses the most balanced 1-2 line split so long Russian words stay inside the frame."""
    total = len(" ".join(w["word"] for w in words))
    if len(words) <= 1 or total <= max_chars:
        return [list(words)]

    best_score = None
    best_split = None
    for split in range(1, len(words)):
        first = " ".join(w["word"] for w in words[:split])
        second = " ".join(w["word"] for w in words[split:])
        if not second:
            continue
        score = max(len(first), len(second))
        if first.endswith(("-", "—")) or second.startswith(("-", "—")):
            score += 3
        if best_score is None or score < best_score:
            best_score = score
            best_split = split
    if best_split is None:
        return [list(words)]
    return [list(words[:best_split]), list(words[best_split:])]


def _render_karaoke_text(lines, accent_index, preset):
    """ASS text where only the currently spoken word is accent-coloured and pops."""
    counter = 0
    rendered_lines = []
    for line in lines:
        parts = []
        for word in line:
            text = word["word"].upper()
            if counter == accent_index:
                pop = "\\fscx112\\fscy112\\t(0,110,\\fscx100\\fscy100)" if preset["pop"] else ""
                parts.append("{\\1c%s%s}%s{\\1c%s}" % (preset["accent"], pop, text, preset["primary"]))
            else:
                parts.append(text)
            counter += 1
        rendered_lines.append(" ".join(parts))
    return "\\N".join(rendered_lines)


def create_ass_subtitles(words, start_t, end_t, output_ass_path, margin_v=420, target_w=1080,
                         target_h=1920, style_name=None, font_size=None):
    """Builds a karaoke ASS file: 1-2 short lines per phrase, the spoken word highlighted."""
    preset = SUBTITLE_PRESETS.get(style_name) or SUBTITLE_PRESETS[DEFAULT_SUBTITLE_STYLE]

    clip_words = [w for w in words if w["end"] > start_t and w["start"] < end_t]

    scale_factor = target_h / 1920.0
    base_font_size = float(font_size) if font_size else 68.0
    scaled_fontsize = max(24, int(round(base_font_size * scale_factor)))
    scaled_margin_v = max(20, int(round(float(margin_v) * scale_factor)))

    # ~0.6 em per uppercase Cyrillic glyph keeps a line inside the frame without touching edges
    usable_width = max(200.0, target_w - 2 * max(12.0, 24.0 * scale_factor))
    max_chars = int(usable_width / max(8.0, 0.60 * scaled_fontsize))
    max_chars = max(12, min(22, max_chars))

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{preset["font"]},{scaled_fontsize},{preset["primary"]},{preset["primary"]},&H00000000,&H50000000,-1,0,0,0,100,100,0,0,1,{preset["outline_width"]},{preset["shadow"]},2,30,30,{scaled_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    clip_duration = end_t - start_t
    events = []

    for phrase in _group_subtitle_phrases(clip_words, max_chars):
        lines = _wrap_phrase_lines(phrase, max_chars)
        for index, word in enumerate(phrase):
            seg_start = max(0.0, word["start"] - start_t)
            if index + 1 < len(phrase):
                seg_end = phrase[index + 1]["start"] - start_t
            else:
                seg_end = phrase[-1]["end"] - start_t
            seg_end = min(clip_duration, seg_end)
            if seg_end <= seg_start:
                seg_end = min(clip_duration, seg_start + 0.12)
            if seg_end <= 0:
                continue
            text = _render_karaoke_text(lines, index, preset)
            events.append(
                f"Dialogue: 0,{format_ass_timestamp(seg_start)},{format_ass_timestamp(seg_end)},Default,,0,0,0,,{text}"
            )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(events))


def measure_loudness(video_path: str, start_t: float, end_t: float):
    """First loudnorm pass: measures the clip's audio so the second pass can hit -14 LUFS."""
    try:
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = CREATE_NO_WINDOW
        result = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-nostdin",
             "-ss", f"{start_t:.3f}", "-t", f"{max(0.1, end_t - start_t):.3f}",
             "-i", video_path, "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180, **kwargs)
        text = result.stderr.decode("utf-8", "replace")
        start = text.rfind("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        data = json.loads(text[start:end + 1])
        return {
            "i": float(data["input_i"]),
            "tp": float(data["input_tp"]),
            "lra": float(data["input_lra"]),
            "thresh": float(data["input_thresh"]),
            "offset": float(data.get("target_offset", 0.0)),
        }
    except Exception:
        return None


def build_loudnorm_filter(measured) -> str:
    """Two-pass loudnorm with measured values; falls back to the plain single-pass filter."""
    if not measured:
        return AUDIO_LOUDNORM
    return ("loudnorm=I=-14:TP=-1.5:LRA=11"
            f":measured_I={measured['i']}:measured_TP={measured['tp']}"
            f":measured_LRA={measured['lra']}:measured_thresh={measured['thresh']}"
            f":offset={measured['offset']}"
            ":linear=true:print_format=none")


def render_clip_ffmpeg(
    video_path: str,
    start_t: float,
    end_t: float,
    ass_path: str,
    output_mp4: str,
    clip_format: str = "9:16",
    crop_offset: str = "Авто-трекинг",
    layout_mode: str = "Обычный кроп",
    webcam_pos: str = "Верхний левый угол",
    webcam_height_pct: float = 30.0,
    webcam_zoom: float = 1.35,
    webcam_offset_x: float = 0.0,
    webcam_offset_y: float = 0.0,
    webcam_face_center: bool = False,
    progress_cb=None
):
    """Renders formatted clip with optional horizontal tracking, streamer webcam layout, and burned-in ASS subtitles using FFmpeg."""
    escaped_ass = escape_ffmpeg_path(ass_path)
    target_w, target_h = FORMAT_RESOLUTIONS.get(clip_format, (1080, 1920))
    # two-pass loudness normalisation (-14 LUFS), only when the source actually has audio
    audio_args = []
    if probe_has_audio(video_path):
        audio_args = ["-af", build_loudnorm_filter(measure_loudness(video_path, start_t, end_t))]

    if crop_offset == "Авто-трекинг" and clip_format != "16:9":
        x_expr = compute_horizontal_tracking_expression(video_path, start_t, end_t)
    elif crop_offset == "Левее":
        x_expr = "(iw-ow)*0.1"
    elif crop_offset == "Правее":
        x_expr = "(iw-ow)*0.9"
    else:
        x_expr = "(iw-ow)/2"

    use_webcam_layout = layout_mode == "Вебка сверху"
    src_size = None
    face_point = None
    webcam_h = 0
    if use_webcam_layout:
        webcam_h = int(round(target_h * (float(webcam_height_pct) / 100.0)))
        src_size = probe_video_size(video_path)
        if webcam_face_center:
            face_point = find_webcam_face(video_path, src_size, webcam_pos, target_w, webcam_h,
                                          start_t, end_t)
            if face_point is None and face_detection_available():
                # No streamer face anywhere in the webcam zone during this clip (fullscreen
                # insert, ad break, cut-away): a plain crop beats an empty piece of interface.
                use_webcam_layout = False

    if use_webcam_layout:
        main_h = target_h - webcam_h
        
        cam_ar = target_w / float(webcam_h)
        main_ar = target_w / float(main_h)

        crop = webcam_crop_box(src_size, webcam_pos, target_w, webcam_h,
                               webcam_zoom, webcam_offset_x, webcam_offset_y, face_point)

        if crop:
            cam_x, cam_y, cam_w, cam_h = crop
            webcam_crop = (f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},"
                           f"scale={target_w}:{webcam_h},setsar=1[webcam]")
        else:
            # fallback when the source resolution could not be probed
            if webcam_pos == "Верхний правый угол":
                cam_x_expr, cam_y_expr = "iw-ow", "0"
            elif webcam_pos == "Нижний левый угол":
                cam_x_expr, cam_y_expr = "0", "ih-oh"
            elif webcam_pos == "Нижний правый угол":
                cam_x_expr, cam_y_expr = "iw-ow", "ih-oh"
            elif "центр" in webcam_pos.lower():
                cam_x_expr = "0" if "лев" in webcam_pos.lower() else "iw-ow"
                cam_y_expr = "(ih-oh)/2"
            else:  # Upper Left
                cam_x_expr, cam_y_expr = "0", "0"
            webcam_crop = (f"[0:v]crop=w='min(iw*{WEBCAM_BASE_FRACTION}, ih*{WEBCAM_BASE_FRACTION}*{cam_ar:.4f})':"
                           f"h='min(ih*{WEBCAM_BASE_FRACTION}, iw*{WEBCAM_BASE_FRACTION}/{cam_ar:.4f})':"
                           f"x='{cam_x_expr}':y='{cam_y_expr}',scale={target_w}:{webcam_h},setsar=1[webcam]")

        main_crop = f"[0:v]crop=w='min(iw, ih*{main_ar:.4f})':h='min(ih, iw/{main_ar:.4f})':x='{x_expr}':y='(ih-oh)/2',scale={target_w}:{main_h},setsar=1[main]"
        vstack_subs = f"[webcam][main]vstack=inputs=2,setsar=1[stacked];[stacked]subtitles='{escaped_ass}'[v]"

        filter_complex_str = f"{webcam_crop};{main_crop};{vstack_subs}"

        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start_t:.3f}",
            "-to", f"{end_t:.3f}",
            "-i", video_path,
            "-filter_complex", filter_complex_str,
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
        ] + audio_args + OUTPUT_VIDEO_FLAGS + [output_mp4]
    else:
        if clip_format == "9:16":
            crop_scale_filter = f"crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)':x='{x_expr}':y='(ih-oh)/2',scale=1080:1920,setsar=1"
        elif clip_format == "4:5":
            crop_scale_filter = f"crop=w='min(iw,ih*4/5)':h='min(ih,iw*5/4)':x='{x_expr}':y='(ih-oh)/2',scale=1080:1350,setsar=1"
        elif clip_format == "1:1":
            crop_scale_filter = f"crop=w='min(iw,ih)':h='min(ih,iw)':x='{x_expr}':y='(ih-oh)/2',scale=1080:1080,setsar=1"
        elif clip_format == "16:9":
            crop_scale_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(1920-iw)/2:(1080-ih)/2,setsar=1"
        else:
            crop_scale_filter = f"crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)':x='{x_expr}':y='(ih-oh)/2',scale={target_w}:{target_h},setsar=1"

        vf_filter = f"{crop_scale_filter},subtitles='{escaped_ass}'"
        
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start_t:.3f}",
            "-to", f"{end_t:.3f}",
            "-i", video_path,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
        ] + audio_args + OUTPUT_VIDEO_FLAGS + [output_mp4]

    run_ffmpeg_with_progress(cmd, end_t - start_t, progress_cb=progress_cb)


def get_unique_output_path(output_dir: str, prefix: str, extension: str = ".mp4") -> str:
    """Generates a guaranteed unique file path using timestamp and sequence counter to prevent file lock conflicts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{timestamp}"
    candidate = os.path.join(output_dir, f"{base_name}{extension}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{base_name}_{counter}{extension}")
        counter += 1
    return candidate


def process_video(
    video_file,
    local_path_input,
    juicy_threshold,
    target_length,
    margin_v,
    max_clips,
    range_start_input,
    range_end_input,
    clip_format,
    crop_offset,
    layout_mode,
    webcam_pos,
    webcam_height_pct,
    webcam_zoom,
    webcam_offset_x,
    webcam_offset_y,
    webcam_face_center,
    subtitle_style,
    subtitle_font_size,
    use_llm,
    llm_provider,
    llm_model,
    llm_base_url,
    llm_key_input,
    progress=gr.Progress()
):
    """Main pipeline handling video processing step by step with optional manual time range restriction.

    This is a generator: it yields (html_feed, files) right after every rendered clip, so
    finished clips show up in the UI one by one instead of only after the whole batch.
    """
    STOP_EVENT.clear()
    
    raw_path, source_error = resolve_video_source(video_file, local_path_input)
    if source_error:
        yield source_error, []
        return

    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix="opus_run_")
    results = []
    llm_status = ""

    def build_feed(clip_results):
        return create_clips_feed_html(clip_results, clip_format=clip_format, status=llm_status)

    def collect_files(clip_results):
        return [r["path"] for r in clip_results if r["path"].startswith(output_dir)]

    try:
        working_video = isolate_and_copy_video(raw_path, temp_dir)

        if STOP_EVENT.is_set():
            yield "🛑 Бурмалда остановлена пользователем.", []
            return

        # Step 1: Parse manual time range if provided
        start_text = str(range_start_input or "").strip()
        end_text = str(range_end_input or "").strip()
        raw_start = parse_time_value(range_start_input)
        raw_end = parse_time_value(range_end_input)

        if raw_start is None or raw_end is None:
            bad_value = range_start_input if raw_start is None else range_end_input
            yield (f"⚠️ Не удалось разобрать время «{str(bad_value).strip()}». "
                   f"Форматы: 95, 1:35 или 01:02:03."), []
            return

        if start_text and end_text and raw_end <= raw_start:
            yield "⚠️ Конец интервала должен быть больше начала.", []
            return

        r_start = raw_start
        r_end = raw_end
        has_manual_range = r_end > r_start
        extract_start = r_start if has_manual_range else 0.0
        extract_duration = (r_end - r_start) if has_manual_range else None

        # Step 2: Analysis — restored from .cache/ when this source + range was seen before
        progress(0.1, desc="Анализ исходника...")
        fingerprint = source_fingerprint(raw_path)
        cached = load_analysis_cache(fingerprint, extract_start, extract_duration)
        if cached is None:
            temp_wav = os.path.join(temp_dir, "audio.wav")
            extract_audio(working_video, temp_wav, start_time=extract_start, duration=extract_duration)

            if STOP_EVENT.is_set():
                yield "🛑 Бурмалда остановлена пользователем.", []
                return

            visual_motion_per_sec = compute_visual_motion_per_second(working_video, start_time=extract_start, duration=extract_duration)

            if STOP_EVENT.is_set():
                yield "🛑 Бурмалда остановлена пользователем.", []
                return

            # Step 3: RMS Volume Analysis & Transcription
            progress(0.3, desc="Транскрибация и анализ динамики...")
            rms_per_sec = compute_rms_per_second(temp_wav)
            words, segments, duration = transcribe_audio(temp_wav)

            if STOP_EVENT.is_set():
                yield "🛑 Бурмалда остановлена пользователем.", []
                return

            save_analysis_cache(fingerprint, extract_start, extract_duration,
                                words, segments, rms_per_sec, visual_motion_per_sec, duration)
        else:
            words, segments, rms_per_sec, visual_motion_per_sec, duration = cached
            progress(0.55, desc="Анализ восстановлен из кэша")

        if has_manual_range:
            for w in words:
                w["start"] += r_start
                w["end"] += r_start
            for s in segments:
                s["start"] += r_start
                s["end"] += r_start
            # The analysed audio can be shorter than the requested range (for example
            # when the range end lies past the end of the file), so never trust r_end blindly.
            effective_duration = min(r_end, r_start + duration) if duration > 0 else r_end
        else:
            effective_duration = duration

        # Step 4: Find juicy moments — sentence-aligned + LLM when available, math otherwise
        progress(0.6, desc="Поиск сочных моментов...")
        range_min_limit = r_start if has_manual_range else 0.0
        range_max_limit = effective_duration

        def math_clips():
            found = find_juicy_clips(
                words=words,
                segments=segments,
                rms_per_sec=rms_per_sec,
                visual_motion_per_sec=visual_motion_per_sec,
                total_duration=effective_duration,
                juicy_threshold=juicy_threshold,
                target_len=target_length,
                max_clips=max_clips,
                range_min=range_min_limit,
                range_max=range_max_limit,
                time_offset=extract_start
            )
            if has_manual_range:
                found = [c for c in found if c["start"] >= r_start and c["end"] <= effective_duration]
            return found

        clips_meta = []
        llm_ok = False
        provider_name = (llm_provider or DEFAULT_LLM_PROVIDER).strip()
        preset = llm_provider_preset(provider_name)
        base_url = (llm_base_url or "").strip() or preset["base_url"]
        model = (llm_model or "").strip() or preset["model"]
        api_key = llm_api_key(provider_name, llm_key_input)
        llm_label = "%s / %s" % (provider_name.split(" — ")[0], model)
        local_endpoint = _is_local_endpoint(base_url)

        if use_llm and not api_key and not local_endpoint:
            llm_status = ("⚠️ LLM-отбор включён, но нет API-ключа для «%s» — "
                          "работает математический скоринг." % provider_name.split(" — ")[0])
            print("[llm] " + llm_status)

        if use_llm and (api_key or local_endpoint):
            progress(0.62, desc="LLM-отбор моментов (%s)..." % llm_label)
            sentence_clips = find_sentence_clips(
                words=words,
                segments=segments,
                target_len=target_length,
                range_min=range_min_limit,
                range_max=range_max_limit,
                time_offset=extract_start
            )
            for cand in sentence_clips:
                cand["score"] = score_clip_candidate(
                    words, rms_per_sec, visual_motion_per_sec,
                    cand["start"], cand["end"], effective_duration, time_offset=extract_start)
            sentence_clips.sort(key=lambda c: c["score"], reverse=True)
            shortlist = sentence_clips[:LLM_MAX_CANDIDATES]

            if shortlist:
                summary = apply_llm_scores(
                    {"base_url": base_url, "api_key": api_key, "model": model,
                     "label": llm_label, "local": local_endpoint},
                    shortlist)
            else:
                summary = {"ok": False, "evaluated": 0, "from_cache": 0, "requested": 0,
                           "ads": 0, "prefiltered": 0, "skipped": 0,
                           "error": "нет кандидатов с текстом"}

            if summary["ok"]:
                llm_ok = True
                # is_ad fragments are dropped outright; the rest keep the LLM score and title
                clips_meta = [c for c in shortlist if not (c.get("llm") or {}).get("is_ad")]
                # a fragment the model never scored is dropped too — an unchecked ad must not
                # slip into the clips just because math scoring liked it
                if LLM_DROP_UNEVALUATED:
                    clips_meta = [c for c in clips_meta if c.get("llm")]
                for cand in clips_meta:
                    verdict = cand.get("llm") or {}
                    if verdict:
                        cand["score"] = verdict.get("score", cand["score"])
                        cand["title"] = verdict.get("title", "")
                clips_meta.sort(key=lambda c: c["score"], reverse=True)

                # sentence windows start on every segment, so the best picks often overlap:
                # keep the highest-scoring ones and skip anything that intersects them
                try:
                    clip_limit = int(max_clips or 0)
                except (TypeError, ValueError):
                    clip_limit = 0
                picked = []
                for cand in clips_meta:
                    if clip_limit > 0 and len(picked) >= clip_limit:
                        break
                    if any(not (cand["end"] <= other["start"] or cand["start"] >= other["end"])
                           for other in picked):
                        continue
                    picked.append(cand)
                clips_meta = picked
                clips_meta.sort(key=lambda c: c["start"])
                for cand in clips_meta:
                    cand.pop("llm", None)

                llm_status = (
                    "🤖 " + llm_label + ": оценено %d из %d кандидатов (из кэша %d), "
                    "рекламы отсеяно: %d (из них по стоп-фразам %d), без оценки: %d, "
                    "в клипы ушло: %d."
                    % (summary["evaluated"], len(shortlist), summary["from_cache"],
                       summary["ads"], summary["prefiltered"], summary["skipped"],
                       len(clips_meta))
                )
            else:
                llm_status = ("⚠️ LLM-отбор недоступен (%s) — включён математический скоринг."
                              % summary["error"])

            progress(0.66, desc=llm_status[:120])
            print("[llm] " + llm_status)

        if not clips_meta and not llm_ok:
            clips_meta = math_clips()

        if not clips_meta and llm_ok:
            yield (llm_status + "<br>Все кандидаты отсеяны как рекламные вставки. "
                   "Отключите LLM-отбор или расширьте интервал."), []
            return

        if not clips_meta:
            yield f"❌ Не найдено моментов с рейтингом выше {juicy_threshold} в заданном диапазоне. Попробуйте снизить порог или изменить интервал.", []
            return

        # Step 5: Render Clips with Unique Filenames inside output/ directory ONLY
        total_found = len(clips_meta)
        target_w, target_h = FORMAT_RESOLUTIONS.get(clip_format, (1080, 1920))

        for idx, clip_info in enumerate(clips_meta, start=1):
            if STOP_EVENT.is_set():
                break

            def clip_progress(fraction, idx=idx, total=total_found, score=clip_info["score"]):
                done = (idx - 1 + max(0.0, min(1.0, fraction))) / max(1, total)
                progress(min(0.95, 0.6 + 0.35 * done),
                         desc=f"Рендер клипа {idx}/{total} (Score: {score}) — {int(fraction * 100)}%")

            clip_progress(0.0)

            ass_file = os.path.join(temp_dir, f"clip_{idx}.ass")
            create_ass_subtitles(
                words=words,
                start_t=clip_info["start"],
                end_t=clip_info["end"],
                output_ass_path=ass_file,
                margin_v=margin_v,
                target_w=target_w,
                target_h=target_h,
                style_name=subtitle_style,
                font_size=subtitle_font_size
            )

            prefix = f"clip_{idx}_score{clip_info['score']}"
            if clip_info.get("title"):
                title_slug = _ascii_slug(clip_info["title"])
                if title_slug:
                    prefix += f"_{title_slug}"
            out_path = get_unique_output_path(output_dir, prefix=prefix, extension=".mp4")

            try:
                render_clip_ffmpeg(
                    video_path=working_video,
                    start_t=clip_info["start"],
                    end_t=clip_info["end"],
                    ass_path=ass_file,
                    output_mp4=out_path,
                    clip_format=clip_format,
                    crop_offset=crop_offset,
                    layout_mode=layout_mode,
                    webcam_pos=webcam_pos,
                    webcam_height_pct=webcam_height_pct,
                    webcam_zoom=webcam_zoom,
                    webcam_offset_x=webcam_offset_x,
                    webcam_offset_y=webcam_offset_y,
                    webcam_face_center=webcam_face_center,
                    progress_cb=clip_progress
                )

                results.append({
                    "path": out_path,
                    "score": clip_info["score"],
                    "index": idx,
                    "duration": round(clip_info["duration"], 1),
                    "title": clip_info.get("title", "")
                })

                # Push the finished clip into the feed immediately (live progress)
                yield build_feed(results), collect_files(results)
            except RuntimeError as re:
                if "Бурмалда остановлена" in str(re):
                    break
                else:
                    raise re

        status_prefix = "🛑 Бурмалда остановлена пользователем." if STOP_EVENT.is_set() else "Готово!"
        progress(1.0, desc=status_prefix)

        yield build_feed(results), collect_files(results)

    except PermissionError as pe:
        yield f"⚠️ Файл заблокирован операционной системой: {str(pe)}. Попробуйте ещё раз через секунду.", []
    except Exception as e:
        if "остановлена пользователем" in str(e):
            yield f"🛑 Бурмалда остановлена пользователем.<br>{build_feed(results)}", collect_files(results)
        else:
            yield f"❌ Ошибка обработки: {str(e)}<br>{build_feed(results)}", collect_files(results)
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def create_clips_feed_html(clips, clip_format: str = "9:16", status: str = ""):
    """Generates HTML snippet for displaying video cards with ratings and download links."""
    cards_html = []
    if status:
        cards_html.append(
            '<div style="background:#1e293b; border:1px solid #334155; border-left:4px solid #6366f1; '
            'border-radius:10px; padding:12px 16px; margin-bottom:16px; color:#e2e8f0; font-size:14px; '
            'line-height:1.45;">' + status + '</div>'
        )
    cards_html.append('<div style="display: flex; flex-direction: column; gap: 20px;">')
    current_time_ts = int(time.time())

    aspect_styles = {
        "9:16": "max-height: 480px; aspect-ratio: 9/16;",
        "4:5": "max-height: 440px; aspect-ratio: 4/5;",
        "1:1": "max-height: 400px; aspect-ratio: 1/1;",
        "16:9": "max-height: 360px; aspect-ratio: 16/9;"
    }
    v_style = aspect_styles.get(clip_format, "max-height: 480px; aspect-ratio: 9/16;")

    for clip in clips:
        rel_path = f"/file={clip['path']}?t={current_time_ts}"
        filename = os.path.basename(clip['path'])
        clip_title = html.escape(str(clip.get("title") or "").strip())
        label = f"🎬 Клип #{clip['index']}" + (f" — {clip_title}" if clip_title else "") + f" ({clip['duration']} sec)"
        cards_html.append(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; color: #f8fafc; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 700; font-size: 16px;">{label}</span>
                <span style="background: linear-gradient(135deg, #6366f1, #a855f7); color: white; padding: 6px 14px; border-radius: 20px; font-weight: 800; font-size: 14px;">
                    Score: {clip['score']}/100
                </span>
            </div>
            <div style="width: 100%; border-radius: 8px; overflow: hidden; background: #0f172a; display: flex; justify-content: center;">
                <video src="{rel_path}" controls style="{v_style} width: auto; border-radius: 8px;"></video>
            </div>
            <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
                <a href="{rel_path}" download="{filename}" 
                   style="background: #2563eb; color: white; padding: 8px 16px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px; display: inline-flex; align-items: center; gap: 6px;">
                    📥 Скачать MP4
                </a>
            </div>
        </div>
        """)
    cards_html.append('</div>')
    return "".join(cards_html)


CUSTOM_CSS = """
body, .gradio-container {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.accent-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.accent-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px -4px rgba(59, 130, 246, 0.5) !important;
}
.panel-box {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks(title="Opus Clip Local — Автоматическая нарезка клипов") as demo:
    gr.Markdown(
        """
        # 🎬 Opus Clip Local
        ### Нарезка длинных видео на вирусящиеся клипы с AI субтитрами
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            face_ok = face_detection_available()
            video_input = gr.File(
                file_count="single",
                file_types=[".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi", ".mpg", ".mpeg", ".ts"],
                type="filepath",
                label="Загрузите видео (MP4 / MKV / MOV)",
                interactive=True
            )
            local_path_input = gr.Textbox(
                value="",
                label="…или полный путь к файлу на диске",
                placeholder=r"C:\Video\запись.mp4",
                info="Для больших файлов надёжнее и быстрее: без копирования через браузер и без блокировок в Temp"
            )
            with gr.Accordion("⚙️ Настройки нарезки", open=True):
                with gr.Row():
                    clip_format = gr.Dropdown(
                        choices=["9:16", "4:5", "1:1", "16:9"],
                        value="9:16",
                        label="Формат клипа",
                        info="Соотношение сторон"
                    )
                    crop_offset = gr.Dropdown(
                        choices=["Авто-трекинг", "Центр", "Левее", "Правее"],
                        value="Авто-трекинг",
                        label="Позиция обрезки (Crop Offset)",
                        info="Смещение или авто-трекинг"
                    )
                with gr.Row():
                    layout_mode = gr.Dropdown(
                        choices=["Обычный кроп", "Вебка сверху"],
                        value="Обычный кроп",
                        label="Режим компоновки кадра",
                        info="Стиль размещения контента"
                    )
                    webcam_pos = gr.Dropdown(
                        choices=["Верхний левый угол", "Верхний правый угол", "Нижний левый угол",
                                 "Нижний правый угол", "По центру слева", "По центру справа"],
                        value="Верхний левый угол",
                        label="Позиция веб-камеры в исходнике",
                        info="Угол или центральная зона (треть экрана) — откуда вырезать вебку"
                    )
                webcam_height_pct = gr.Slider(
                    minimum=20,
                    maximum=40,
                    value=30,
                    step=5,
                    label="Высота веб-камеры (%)",
                    info="Доля высоты итогового видео"
                )
                with gr.Row():
                    webcam_zoom = gr.Slider(
                        minimum=1.0,
                        maximum=2.2,
                        value=1.35,
                        step=0.05,
                        label="Зум веб-камеры",
                        info="Больше — плотнее рамка вокруг лица, меньше фона и рамок браузера"
                    )
                    webcam_face_center = gr.Checkbox(
                        value=False,
                        label="Автоцентр вебки по лицу",
                        info=("YuNet-детектор OpenCV сам наводит рамку на стримера" if face_ok
                              else "OpenCV не найден — автоцентр недоступен")
                    )
                with gr.Row():
                    webcam_offset_x = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1,
                        label="Смещение вебки по X (%)",
                        info="0 — вплотную к краю кадра, 100 — максимально к центру"
                    )
                    webcam_offset_y = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1,
                        label="Смещение вебки по Y (%)",
                        info="0 — к верхней кромке зоны, 100 — к нижней (для «по центру» зона = 20–80 % высоты)"
                    )
                juicy_threshold = gr.Slider(
                    minimum=50,
                    maximum=95,
                    value=75,
                    step=1,
                    label="Порог сочности (Score min threshold)",
                    info="Забирает все моменты выше этого балла"
                )
                target_length = gr.Slider(
                    minimum=20,
                    maximum=60,
                    value=40,
                    step=5,
                    label="Желаемая длина клипа (сек)",
                    info="Длина нарезанных отрезков"
                )
                margin_v = gr.Number(
                    value=420,
                    label="Отступ субтитров снизу MarginV (px)",
                    info="Оставляет чистое место под баннеры"
                )
                with gr.Row():
                    subtitle_style = gr.Dropdown(
                        choices=list(SUBTITLE_PRESETS.keys()),
                        value=DEFAULT_SUBTITLE_STYLE,
                        label="Стиль субтитров",
                        info="Акцентное слово подсвечивается и «подпрыгивает»"
                    )
                    subtitle_font_size = gr.Slider(
                        minimum=40,
                        maximum=110,
                        value=68,
                        step=2,
                        label="Размер шрифта субтитров",
                        info="Пиксели при ширине 1080"
                    )
                max_clips = gr.Number(
                    value=0,
                    precision=0,
                    label="Максимум клипов (0 = без ограничений)",
                    info="Лимит количества клипов"
                )
                with gr.Row():
                    range_start_input = gr.Textbox(
                        value="",
                        label="Начало интервала (сек или MM:SS)",
                        placeholder="0 или 01:20",
                        info="Необязательно"
                    )
                    range_end_input = gr.Textbox(
                        value="",
                        label="Конец интервала (сек или MM:SS)",
                        placeholder="600 или 15:00",
                        info="Необязательно"
                    )

            with gr.Accordion("🤖 LLM-отбор моментов (OpenAI-совместимый)", open=False):
                llm_provider = gr.Dropdown(
                    choices=list(LLM_PROVIDERS.keys()),
                    value=DEFAULT_LLM_PROVIDER,
                    label="Провайдер LLM",
                    info="Groq / Cerebras — бесплатные тарифы, Ollama — локально и без лимитов"
                )
                with gr.Row():
                    llm_model = gr.Textbox(
                        value=LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]["model"],
                        label="Модель",
                        info="Любой id модели у выбранного провайдера"
                    )
                    llm_key_input = gr.Textbox(
                        value=os.environ.get(LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]["key_env"], ""),
                        type="password",
                        label="API Key провайдера",
                        info="Для Ollama не нужен; можно задать в .env (GROQ_API_KEY и т.п.)"
                    )
                llm_base_url = gr.Textbox(
                    value=LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]["base_url"],
                    label="Base URL (OpenAI-совместимый)",
                    info="Подставляется автоматически при смене провайдера"
                )
                use_llm = gr.Checkbox(
                    value=bool(llm_api_key(DEFAULT_LLM_PROVIDER)),
                    label="Умный LLM-отбор моментов",
                    info="Один запрос на ролик: отсев рекламы, оценка хука, заголовки клипов"
                )
                llm_provider.change(
                    fn=llm_provider_fields,
                    inputs=[llm_provider],
                    outputs=[llm_model, llm_base_url, llm_key_input]
                )

            with gr.Row():
                cut_btn = gr.Button("⚡ Нарезать клипы", variant="primary", elem_classes=["accent-btn"])
                stop_btn = gr.Button("🛑 остановить бурмалду", variant="stop")

        with gr.Column(scale=1):
            gr.Markdown("### 🔥 Готовые вертикальные клипы")
            clips_feed = gr.HTML(
                value='<div style="color: #94a3b8; text-align: center; padding: 40px;">Загрузите видео и нажмите «Нарезать клипы»</div>'
            )
            file_output = gr.Files(label="Файлы для скачивания", visible=True)

    cut_event = cut_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            local_path_input,
            juicy_threshold,
            target_length,
            margin_v,
            max_clips,
            range_start_input,
            range_end_input,
            clip_format,
            crop_offset,
            layout_mode,
            webcam_pos,
            webcam_height_pct,
            webcam_zoom,
            webcam_offset_x,
            webcam_offset_y,
            webcam_face_center,
            subtitle_style,
            subtitle_font_size,
            use_llm,
            llm_provider,
            llm_model,
            llm_base_url,
            llm_key_input
        ],
        outputs=[clips_feed, file_output]
    )

    stop_btn.click(
        fn=stop_processing,
        inputs=[],
        outputs=[],
        cancels=[cut_event]
    )

if __name__ == "__main__":
    output_directory = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_directory, exist_ok=True)

    # Housekeeping: Gradio copies uploads and served files into GRADIO_TEMP_DIR, so a 2 GB
    # source leaves a 2 GB copy behind on every upload. Purge anything older than a day.
    removed_uploads = cleanup_old_files(GRADIO_TEMP_DIR, UPLOAD_MAX_AGE_HOURS)
    if removed_uploads:
        print(f"Убрано старых файлов из gradio_tmp: {removed_uploads}")

    removed_cache = cleanup_old_files(CACHE_DIR, CACHE_MAX_AGE_DAYS * 24)
    if removed_cache:
        print(f"Убрано старых файлов из .cache: {removed_cache}")

    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "allowed_paths": [output_directory],
    }
    # Gradio 6 moved the custom CSS from Blocks(...) to launch(...); older 4.x builds
    # do not accept it there, so pass it only when the signature supports it.
    if "css" in inspect.signature(demo.launch).parameters:
        launch_kwargs["css"] = CUSTOM_CSS

    demo.queue().launch(**launch_kwargs)
