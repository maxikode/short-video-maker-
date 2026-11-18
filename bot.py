import asyncio
import base64
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    FSInputFile, Update
)
from aiogram.client.default import DefaultBotProperties
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from groq import Groq
import yt_dlp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("short-video-maker")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-20b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
PORT = int(os.getenv("PORT", "10000"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "250"))
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "90"))
MAX_CLIPS = 10
MIN_CLIP_SECONDS = 12
MAX_CLIP_SECONDS = 60
SAMPLE_INTERVAL = 8
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/short-video-maker"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not BOT_TOKEN or not GROQ_API_KEY:
    raise RuntimeError("Set BOT_TOKEN and GROQ_API_KEY environment variables.")

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
groq = Groq(api_key=GROQ_API_KEY, timeout=120.0, max_retries=2)

# Settings are kept in memory. On a Render restart defaults are restored.
# A persistent database can be added later if settings need to survive restarts.
user_settings = {}
active_users = set()


def settings_for(user_id: int):
    return user_settings.setdefault(user_id, {"subtitles": True, "count": 3})


def settings_keyboard(user_id: int):
    s = settings_for(user_id)
    sub = "✅ С субтитрами" if s["subtitles"] else "⬜ Без субтитров"
    rows = [[InlineKeyboardButton(text=sub, callback_data="toggle_sub")]]
    rows.append([
        InlineKeyboardButton(text=("✅ " if s["count"] == n else "") + str(n), callback_data=f"count_{n}")
        for n in (1, 3, 5, 10)
    ])
    rows.append([InlineKeyboardButton(text="Готово", callback_data="settings_done")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def start_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
        [InlineKeyboardButton(text="ℹ️ Как пользоваться", callback_data="help")]
    ])


def safe_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Пришли обычную публичную ссылку, начинающуюся с https://")
    host = parsed.hostname.lower()
    if host in ("localhost",) or host.endswith((".local", ".internal")):
        raise ValueError("Локальные и внутренние адреса не поддерживаются.")
    try:
        ip = ipaddress.ip_address(host)
        if not ip.is_global:
            raise ValueError("Частные и локальные IP-адреса не поддерживаются.")
    except ValueError as e:
        if "не поддерживаются" in str(e):
            raise
    except Exception:
        pass
    # Reject credentials embedded in URLs.
    if parsed.username or parsed.password:
        raise ValueError("Ссылки с логином или паролем не поддерживаются.")
    return url.strip()


def run_ffmpeg(args, timeout=900):
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "FFmpeg error")[-1800:])
    return result


def probe_video(path: Path):
    import subprocess
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-show_entries", "stream=codec_type,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30
    )
    if p.returncode != 0:
        raise RuntimeError("Не удалось прочитать видео. Возможно, формат не поддерживается.")
    data = json.loads(p.stdout)
    duration = float(data.get("format", {}).get("duration", 0))
    if duration <= 0:
        raise RuntimeError("Не удалось определить длительность видео.")
    if duration > MAX_VIDEO_MINUTES * 60:
        raise RuntimeError(f"Видео слишком длинное. Максимум сейчас: {MAX_VIDEO_MINUTES} минут.")
    streams = data.get("streams", [])
    has_audio = any(x.get("codec_type") == "audio" for x in streams)
    return duration, has_audio


def download_url(url: str, output_dir: Path) -> Path:
    url = safe_url(url)
    template = str(output_dir / "source.%(ext)s")
    opts = {
        "outtmpl": template,
        "format": "best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "socket_timeout": 30,
        "max_filesize": MAX_DOWNLOAD_MB * 1024 * 1024,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in output_dir.glob("source.*") if p.is_file() and p.suffix.lower() not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Не удалось скачать видео по этой ссылке. Попробуй отправить сам файл.")
    path = max(files, key=lambda p: p.stat().st_size)
    if path.stat().st_size > MAX_DOWNLOAD_MB * 1024 * 1024:
        raise RuntimeError(f"Файл превышает лимит {MAX_DOWNLOAD_MB} МБ.")
    return path


def extract_frames(video: Path, out_dir: Path, duration: float):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    # Evenly sample the whole timeline; cap API calls and processing time.
    count = min(180, max(1, int(duration // SAMPLE_INTERVAL) + 1))
    interval = max(SAMPLE_INTERVAL, duration / count)
    timestamps = [min(duration - 0.2, i * interval) for i in range(count)]
    paths = []
    for idx, ts in enumerate(timestamps):
        target = frames_dir / f"frame_{idx:04d}.jpg"
        run_ffmpeg(["-ss", f"{ts:.2f}", "-i", str(video), "-frames:v", "1",
                    "-vf", "scale='min(640,iw)':-2", "-q:v", "6", str(target)], timeout=45)
        if target.exists() and target.stat().st_size:
            paths.append((ts, target))
    return paths


def image_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def analyze_frame_batch(batch):
    content = [{
        "type": "text",
        "text": (
            "Ты анализируешь кадры видео для поиска потенциально интересных коротких клипов. "
            "Кадры идут по времени. Для каждого кадра кратко опиши видимое действие, эмоции, "
            "резкую смену ситуации или визуально необычное событие. Не выдумывай звук и речь. "
            "Верни JSON-массив объектов с полями time и observation. time — время кадра в секундах."
        )
    }]
    for ts, path in batch:
        content.append({"type": "text", "text": f"Кадр около {ts:.1f} секунды:"})
        content.append({"type": "image_url", "image_url": {"url": image_data(path)}})
    result = groq.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_completion_tokens=1200,
    )
    raw = result.choices[0].message.content or "[]"
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
        return items if isinstance(items, list) else []
    except json.JSONDecodeError:
        return []


def transcribe_audio(video: Path, out_dir: Path):
    audio = out_dir / "audio.mp3"
    # Low-bitrate mono audio keeps each chunk below Groq's 25 MB Free Tier upload limit.
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                "-b:a", "48k", str(audio)], timeout=600)
    if not audio.exists() or audio.stat().st_size == 0:
        return []
    chunk_seconds = 7 * 60
    duration = probe_audio_duration(audio)
    segments = []
    for start in range(0, int(duration) + 1, chunk_seconds):
        chunk = out_dir / f"audio_{start}.mp3"
        run_ffmpeg(["-ss", str(start), "-i", str(audio), "-t", str(chunk_seconds),
                    "-c", "copy", str(chunk)], timeout=120)
        if not chunk.exists() or chunk.stat().st_size == 0:
            continue
        try:
            with chunk.open("rb") as f:
                tr = groq.audio.transcriptions.create(
                    file=(chunk.name, f.read()),
                    model=GROQ_STT_MODEL,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            for seg in getattr(tr, "segments", []) or []:
                item = dict(seg) if isinstance(seg, dict) else {
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", ""),
                }
                item["start"] = float(item.get("start", 0)) + start
                item["end"] = float(item.get("end", 0)) + start
                if item.get("text"):
                    segments.append(item)
        except Exception as e:
            log.warning("Transcription chunk at %s failed: %s", start, e)
            # Continue with the visual analysis even if one audio chunk fails.
        finally:
            chunk.unlink(missing_ok=True)
    audio.unlink(missing_ok=True)
    return segments


def probe_audio_duration(path: Path):
    import subprocess
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30
    )
    return float(p.stdout.strip() or 0)


def choose_clips(duration: float, transcript, observations, count: int):
    # Keep the prompt compact while preserving timestamps.
    speech = [
        {"start": round(float(x["start"]), 1), "end": round(float(x["end"]), 1),
         "text": str(x["text"])[:350]}
        for x in transcript[:900]
    ]
    visual = [
        {"time": round(float(x.get("time", 0)), 1),
         "observation": str(x.get("observation", ""))[:220]}
        for x in observations[:180]
    ]
    prompt = f"""
Найди до {count} лучших самостоятельных коротких фрагментов из видео длительностью {duration:.1f} секунд.
Выбирай реальные интересные события: юмор, неожиданность, сильная реакция, напряжённый игровой момент,
необычное действие или запоминающаяся фраза. Не называй обычные паузы интересными.
Данные ниже — подсказки с таймкодами, а не полный видеоряд.
Речь: {json.dumps(speech, ensure_ascii=False)}
Кадры: {json.dumps(visual, ensure_ascii=False)}
Верни только JSON-массив, без Markdown. Каждый объект:
{{"start": число, "end": число, "score": целое 1..100, "reason": "короткое объяснение", "title": "короткое название"}}
Правила:
- Каждый клип от 12 до 60 секунд.
- Таймкоды должны быть внутри видео; оставь небольшой контекст до и после события.
- Не перекрывай клипы друг с другом; избегай почти одинаковых моментов.
- Оценивай честно: score — оценка потенциальной интересности, а не гарантия популярности.
- Если хороших моментов меньше {count}, верни меньше, не заполняй список случайными отрывками.
"""
    result = groq.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": "Ты аккуратный видеоредактор. Возвращай только корректный JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=2500,
        response_format={"type": "json_object"},
    )
    raw = result.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        clips = data.get("clips", []) if isinstance(data, dict) else data
    except Exception:
        match = re.search(r"\[[\s\S]*\]", raw)
        clips = json.loads(match.group(0)) if match else []
    cleaned = []
    for item in clips:
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
            if end - start < MIN_CLIP_SECONDS:
                continue
            end = min(end, start + MAX_CLIP_SECONDS)
            if end - start >= MIN_CLIP_SECONDS:
                cleaned.append({
                    "start": start, "end": end,
                    "score": max(1, min(100, int(item.get("score", 50)))),
                    "reason": str(item.get("reason", "Интересный момент"))[:240],
                    "title": str(item.get("title", "Лучший момент"))[:80],
                })
        except (KeyError, TypeError, ValueError):
            continue
    cleaned.sort(key=lambda x: x["score"], reverse=True)
    result = []
    for item in cleaned:
        if all(item["end"] <= old["start"] or item["start"] >= old["end"] for old in result):
            result.append(item)
        if len(result) >= count:
            break
    return result


def make_srt(transcript, start: float, end: float, path: Path):
    def stamp(seconds):
        ms = int(max(0, seconds) * 1000)
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, milli = divmod(rem, 1000)
        return f"{h:02}:{m:02}:{s:02},{milli:03}"
    rows = []
    idx = 1
    for seg in transcript:
        a, b = float(seg["start"]), float(seg["end"])
        if b <= start or a >= end:
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not text:
            continue
        rows.append(f"{idx}\n{stamp(max(0, a-start))} --> {stamp(min(end-start, b-start))}\n{text}\n")
        idx += 1
    path.write_text("\n".join(rows), encoding="utf-8")


def render_clip(video: Path, clip: dict, transcript, out_dir: Path, index: int, subtitles: bool):
    target = out_dir / f"clip_{index:02}.mp4"
    start, end = clip["start"], clip["end"]
    args = ["-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end-start:.3f}"]
    if subtitles:
        srt = out_dir / f"clip_{index:02}.srt"
        make_srt(transcript, start, end, srt)
        if srt.stat().st_size:
            escaped = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            args += ["-vf", f"subtitles='{escaped}':force_style='FontSize=18,Outline=2,Shadow=1,Alignment=2,MarginV=28'"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(target)]
    run_ffmpeg(args, timeout=600)
    return target


async def process_video(message: Message, source: Path, workdir: Path, user_id: int):
    s = settings_for(user_id)
    await message.answer("🔎 Получил видео. Проверяю длительность и анализирую кадры и звук. Это может занять время.")
    try:
        duration, has_audio = await asyncio.to_thread(probe_video, source)
        frames = await asyncio.to_thread(extract_frames, source, workdir, duration)
        observations = []
        for i in range(0, len(frames), 3):
            try:
                batch = frames[i:i+3]
                found = await asyncio.to_thread(analyze_frame_batch, batch)
                observations.extend(found)
            except Exception as e:
                log.warning("Vision batch failed: %s", e)
        transcript = []
        if has_audio:
            status = await message.answer("🎧 Кадры просмотрены. Распознаю речь и ищу подходящие фрагменты…")
            transcript = await asyncio.to_thread(transcribe_audio, source, workdir)
            await status.delete()
        if not observations and not transcript:
            raise RuntimeError("Не удалось получить данные для анализа. Проверь, что Groq поддерживает выбранные модели.")
        clips = await asyncio.to_thread(choose_clips, duration, transcript, observations, s["count"])
        if not clips:
            await message.answer("Не удалось уверенно найти подходящие моменты. Попробуй другое видео или измени модели Groq.")
            return
        await message.answer(f"✂️ Нашёл {len(clips)} моментов. Нарезаю клипы…")
        for idx, clip in enumerate(clips, 1):
            target = await asyncio.to_thread(render_clip, source, clip, transcript, workdir, idx, s["subtitles"])
            caption = (
                f"🎬 <b>{clip['title']}</b>\n"
                f"⭐ Интересность: <b>{clip['score']}/100</b>\n"
                f"⏱ {int(clip['start']//60):02}:{int(clip['start']%60):02} — "
                f"{int(clip['end']//60):02}:{int(clip['end']%60):02}\n"
                f"💡 {clip['reason']}"
            )
            await message.answer_video(FSInputFile(target), caption=caption, supports_streaming=True)
        await message.answer("✅ Готово! Оценка интересности — ориентир ИИ, а не гарантия просмотров.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def accept_source(message: Message, user_id: int, source_path: Path):
    if user_id in active_users:
        await message.answer("⏳ Я уже обрабатываю твоё видео. Дождись окончания.")
        source_path.unlink(missing_ok=True)
        return
    active_users.add(user_id)
    workdir = source_path.parent
    try:
        await process_video(message, source_path, workdir, user_id)
    except Exception as e:
        log.exception("Video processing failed")
        text = str(e)
        if len(text) > 900:
            text = text[-900:]
        await message.answer("❌ Не удалось обработать видео.\n" + text)
    finally:
        active_users.discard(user_id)
        shutil.rmtree(workdir, ignore_errors=True)


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет! 🎬 Я <b>Short Video Maker</b>.\n\n"
        "Отправь видео файлом или публичную ссылку — я проанализирую кадры и речь, "
        "выберу лучшие моменты, нарежу короткие клипы и поставлю каждому оценку интересности.\n\n"
        "Настройки можно изменить кнопкой ниже.",
        reply_markup=start_keyboard()
    )


@dp.callback_query(F.data == "settings")
async def settings_callback(callback: CallbackQuery):
    await callback.message.edit_text("⚙️ <b>Настройки клипов</b>\nВыбери субтитры и количество клипов:", reply_markup=settings_keyboard(callback.from_user.id))
    await callback.answer()


@dp.callback_query(F.data == "toggle_sub")
async def toggle_sub(callback: CallbackQuery):
    s = settings_for(callback.from_user.id)
    s["subtitles"] = not s["subtitles"]
    await callback.message.edit_reply_markup(reply_markup=settings_keyboard(callback.from_user.id))
    await callback.answer("Настройка обновлена")


@dp.callback_query(F.data.startswith("count_"))
async def set_count(callback: CallbackQuery):
    n = int(callback.data.split("_", 1)[1])
    if n not in (1, 3, 5, 10):
        await callback.answer("Недопустимое количество", show_alert=True)
        return
    settings_for(callback.from_user.id)["count"] = n
    await callback.message.edit_reply_markup(reply_markup=settings_keyboard(callback.from_user.id))
    await callback.answer(f"Буду искать до {n} клипов")


@dp.callback_query(F.data == "settings_done")
async def settings_done(callback: CallbackQuery):
    await callback.message.edit_text("✅ Настройки сохранены для этого запуска бота. Отправь видео или ссылку.")
    await callback.answer()


@dp.callback_query(F.data == "help")
async def help_callback(callback: CallbackQuery):
    await callback.message.answer(
        "📌 <b>Как пользоваться</b>\n"
        "1. Отправь видео как файл или пришли публичную ссылку.\n"
        "2. В настройках выбери субтитры и количество клипов.\n"
        "3. Дождись анализа — бот отправит выбранные фрагменты с оценкой 1–100.\n\n"
        f"Ограничения текущей версии: до {MAX_VIDEO_MINUTES} минут и {MAX_DOWNLOAD_MB} МБ для ссылок. "
        "Лимит Telegram на скачивание файлов ботом также действует."
    )
    await callback.answer()


@dp.message(F.video)
async def video_message(message: Message):
    user_id = message.from_user.id
    workdir = Path(tempfile.mkdtemp(prefix="svm_", dir=DATA_DIR))
    path = workdir / "uploaded_video.mp4"
    try:
        file = await bot.get_file(message.video.file_id)
        await bot.download_file(file.file_path, destination=path)
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        await message.answer("Не удалось скачать видео из Telegram. Попробуй отправить его как документ или ссылку.\n" + str(e)[:400])
        return
    if path.stat().st_size > MAX_DOWNLOAD_MB * 1024 * 1024:
        shutil.rmtree(workdir, ignore_errors=True)
        await message.answer(f"Видео больше лимита {MAX_DOWNLOAD_MB} МБ.")
        return
    await accept_source(message, user_id, path)


@dp.message(F.document)
async def document_message(message: Message):
    doc = message.document
    name = (doc.file_name or "").lower()
    if not (name.endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v")) or (doc.mime_type or "").startswith("video/")):
        await message.answer("Пришли видеофайл (MP4, MOV, MKV или WEBM) либо публичную ссылку.")
        return
    if doc.file_size and doc.file_size > MAX_DOWNLOAD_MB * 1024 * 1024:
        await message.answer(f"Файл больше лимита {MAX_DOWNLOAD_MB} МБ.")
        return
    user_id = message.from_user.id
    workdir = Path(tempfile.mkdtemp(prefix="svm_", dir=DATA_DIR))
    path = workdir / (Path(doc.file_name or "video.mp4").name)
    try:
        file = await bot.get_file(doc.file_id)
        await bot.download_file(file.file_path, destination=path)
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        await message.answer("Не удалось скачать файл из Telegram: " + str(e)[:400])
        return
    await accept_source(message, user_id, path)


@dp.message(F.text)
async def text_message(message: Message):
    text = (message.text or "").strip()
    if not text.startswith(("http://", "https://")):
        await message.answer("Отправь видео файлом или публичную ссылку на видео.")
        return
    user_id = message.from_user.id
    if user_id in active_users:
        await message.answer("⏳ Сначала дождись завершения текущего видео.")
        return
    workdir = Path(tempfile.mkdtemp(prefix="svm_", dir=DATA_DIR))
    active_users.add(user_id)
    try:
        await message.answer("⬇️ Скачиваю видео по ссылке…")
        source = await asyncio.to_thread(download_url, text, workdir)
        await process_video(message, source, workdir, user_id)
    except Exception as e:
        log.exception("URL processing failed")
        await message.answer("❌ Не удалось обработать ссылку.\n" + str(e)[-900:])
    finally:
        active_users.discard(user_id)
        shutil.rmtree(workdir, ignore_errors=True)


async def health(request):
    return web.Response(text="OK")


async def on_startup(app):
    await bot.set_webhook(
        f"{WEBHOOK_URL}/telegram/webhook",
        secret_token=WEBHOOK_SECRET or None,
        drop_pending_updates=True
    )
    log.info("Telegram webhook configured")


async def on_shutdown(app):
    await bot.delete_webhook()
    await bot.session.close()


def create_app():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET or None
    ).register(app, path="/telegram/webhook")
    setup_application(app, dp, bot=bot)
    return app


if __name__ == "__main__":
    if not WEBHOOK_URL:
        raise RuntimeError("Set WEBHOOK_URL to your Render service URL, for example https://your-service.onrender.com")
    web.run_app(create_app(), host="0.0.0.0", port=PORT)
