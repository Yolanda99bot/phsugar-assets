#!/usr/bin/env python3
"""PHSUGAR FB广告素材裂变Bot

Features
--------
• Original pixel-level image noise (5 variants)
• Original ffmpeg video noise  (3 variants)
• 自动换背景 – replace near-background pixels with a gradient
• 自动换颜色 – random colour grade (hue shift, brightness, contrast, tint)
• 自动换字体 – brand-text watermark with random style & position
• 自动换字幕 – subtitle bar at the bottom
• TG inline control panel – toggle each feature, select-all / clear, generate
"""

import logging
import os
import random
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

WORK_DIR = Path("./materials")
WORK_DIR.mkdir(exist_ok=True)

# ── generation counts ────────────────────────────────────────────────
IMAGE_VARIANTS = 5
VIDEO_VARIANTS = 3

# ── original noise params ────────────────────────────────────────────
NUM_PIXELS = 50
NOISE_RANGE = 20

# ── feature keys ─────────────────────────────────────────────────────
F_NOISE = "noise"
F_BG = "bg"
F_COLOR = "color"
F_FONT = "font"
F_CAPTION = "caption"

ALL_FEATURES: List[str] = [F_NOISE, F_BG, F_COLOR, F_FONT, F_CAPTION]

FEATURE_LABELS: Dict[str, Tuple[str, str]] = {
    F_NOISE:   ("🔀", "噪点"),
    F_BG:      ("🖼", "换背景"),
    F_COLOR:   ("🎨", "换颜色"),
    F_FONT:    ("✏️", "换字体"),
    F_CAPTION: ("💬", "换字幕"),
}

# ── content pools ─────────────────────────────────────────────────────
CAPTIONS = [
    "立即领取 BONUS",
    "限时优惠",
    "注册送彩金",
    "每日任务",
    "首存红利",
    "周返水活动",
    "邀请好友",
    "VIP专属",
]

# ASCII-only captions used by ffmpeg drawtext (avoids font/encoding issues)
ASCII_CAPTIONS = [
    "CLAIM BONUS NOW",
    "LIMITED OFFER",
    "REGISTER & GET BONUS",
    "DAILY MISSION",
    "FIRST DEPOSIT BONUS",
    "WEEKLY CASHBACK",
    "INVITE FRIENDS",
    "VIP EXCLUSIVE",
]

BRAND_TEXTS = ["PHSUGAR", "PH SUGAR", "PHSugar.com", "* PHSUGAR *"]

ACCENT_COLORS: List[Tuple[int, int, int]] = [
    (255, 220, 60),
    (255, 80, 80),
    (80, 200, 120),
    (80, 160, 255),
    (200, 80, 255),
    (255, 140, 0),
]

BG_GRADIENTS: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
    ((15, 5, 40), (80, 0, 90)),
    ((5, 20, 50), (0, 80, 120)),
    ((50, 5, 5), (160, 30, 0)),
    ((5, 35, 10), (0, 100, 50)),
    ((10, 10, 10), (50, 50, 60)),
    ((40, 20, 0), (120, 80, 0)),
]

SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]


# ═══════════════════════════ utilities ═══════════════════════════════

def _make_gradient(size: Tuple[int, int],
                   c1: Tuple[int, int, int],
                   c2: Tuple[int, int, int]) -> Image.Image:
    """Top-to-bottom linear gradient."""
    w, h = size
    bg = Image.new("RGB", size)
    draw = ImageDraw.Draw(bg)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (w - 1, y)], fill=(r, g, b))
    return bg


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in SYSTEM_FONTS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ═══════════════════════ image mutation steps ═════════════════════════

def _apply_noise(img: Image.Image) -> Image.Image:
    """Original pixel-level random noise."""
    new_img = img.copy()
    px = new_img.load()
    w, h = new_img.size
    for _ in range(NUM_PIXELS):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r, g, b = px[x, y]
        r = max(0, min(255, r + random.randint(-NOISE_RANGE, NOISE_RANGE)))
        g = max(0, min(255, g + random.randint(-NOISE_RANGE, NOISE_RANGE)))
        b = max(0, min(255, b + random.randint(-NOISE_RANGE, NOISE_RANGE)))
        px[x, y] = (r, g, b)
    return new_img


def _apply_bg_swap(img: Image.Image) -> Image.Image:
    """Replace near-background pixels with a random gradient."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    if h < 2 or w < 2:
        return img

    # Estimate background colour by sampling the four corners
    corners = np.stack([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]])
    bg_est = corners.mean(axis=0)

    # Mask pixels close to the estimated background
    diff = np.linalg.norm(arr - bg_est, axis=2)
    threshold = random.randint(45, 80)
    mask = diff < threshold

    c1, c2 = random.choice(BG_GRADIENTS)
    new_bg = np.array(_make_gradient((w, h), c1, c2), dtype=np.float32)

    result = arr.copy()
    result[mask] = new_bg[mask]
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def _apply_color_shift(img: Image.Image) -> Image.Image:
    """Random colour grade: saturation, brightness, contrast + additive tint."""
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 2.0))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.35))

    arr = np.array(img).astype(np.float32)
    tint = np.array([
        random.randint(-25, 25),
        random.randint(-25, 25),
        random.randint(-25, 25),
    ], dtype=np.float32)
    return Image.fromarray(np.clip(arr + tint, 0, 255).astype(np.uint8))


def _apply_font_overlay(img: Image.Image) -> Image.Image:
    """Stamp a brand watermark with randomised style and position."""
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = rgba.size
    font_size = random.randint(max(16, h // 14), max(28, h // 7))
    font = _load_font(font_size)
    text = random.choice(BRAND_TEXTS)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 12

    positions = {
        "TL": (pad, pad),
        "TR": (w - tw - pad, pad),
        "BL": (pad, h - th - pad),
        "BR": (w - tw - pad, h - th - pad),
        "CT": ((w - tw) // 2, (h - th) // 2),
    }
    x, y = random.choice(list(positions.values()))

    color = random.choice(ACCENT_COLORS)
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(*color, 230))

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _apply_caption(img: Image.Image) -> Image.Image:
    """Add a translucent subtitle bar at the bottom of the image."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    bar_h = max(36, h // 7)

    bar = Image.new("RGBA", (w, bar_h), (0, 0, 0, 200))
    draw = ImageDraw.Draw(bar)
    font_size = max(14, bar_h // 2)
    font = _load_font(font_size)
    text = random.choice(CAPTIONS)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = (bar_h - th) // 2
    draw.text((tx, ty), text, font=font, fill=(*random.choice(ACCENT_COLORS), 255))

    rgba.paste(bar, (0, h - bar_h), mask=bar)
    return rgba.convert("RGB")


def generate_image_variants(path: str, features: Dict[str, bool]) -> List[str]:
    """Return IMAGE_VARIANTS mutated images from *path* using *features*."""
    source = Image.open(path).convert("RGB")
    outputs: List[str] = []

    for _ in range(IMAGE_VARIANTS):
        img = source.copy()
        if features.get(F_NOISE, True):
            img = _apply_noise(img)
        if features.get(F_BG, True):
            img = _apply_bg_swap(img)
        if features.get(F_COLOR, True):
            img = _apply_color_shift(img)
        if features.get(F_FONT, True):
            img = _apply_font_overlay(img)
        if features.get(F_CAPTION, True):
            img = _apply_caption(img)

        out = WORK_DIR / f"{uuid.uuid4().hex}.jpg"
        img.save(str(out), quality=95)
        outputs.append(str(out))

    return outputs


# ═══════════════════════ video mutation ══════════════════════════════

def _build_vf(features: Dict[str, bool]) -> str:
    """Compose an ffmpeg -vf filter chain from the active features."""
    parts: List[str] = []

    if features.get(F_NOISE, True):
        parts.append("noise=alls=20:allf=t+u")

    if features.get(F_COLOR, True):
        hue = random.randint(-30, 30)
        br = round(random.uniform(-0.05, 0.1), 3)
        ct = round(random.uniform(0.9, 1.3), 3)
        parts.append(f"hue=h={hue}")
        parts.append(f"eq=brightness={br}:contrast={ct}")

    font_arg = ""
    for fp in SYSTEM_FONTS:
        if os.path.exists(fp):
            font_arg = f":fontfile={fp}"
            break

    if features.get(F_CAPTION, True):
        safe = random.choice(ASCII_CAPTIONS).replace("'", "\\'").replace(":", "\\:")
        parts.append(
            f"drawtext=text='{safe}'"
            f":x=(w-text_w)/2:y=h-60"
            f":fontsize=36:fontcolor=yellow{font_arg}"
            f":box=1:boxcolor=black@0.6:boxborderw=8"
        )

    if features.get(F_FONT, True):
        parts.append(
            f"drawtext=text='PHSUGAR'"
            f":x=10:y=10:fontsize=28:fontcolor=white@0.85{font_arg}"
        )

    return ",".join(parts) if parts else "null"


def generate_video_variants(path: str, features: Dict[str, bool]) -> List[str]:
    """Return VIDEO_VARIANTS mutated videos from *path* using *features*."""
    outputs: List[str] = []

    for _ in range(VIDEO_VARIANTS):
        out = WORK_DIR / f"{uuid.uuid4().hex}.mp4"
        vf = _build_vf(features)

        cmd = [
            "ffmpeg", "-i", path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-y", str(out),
        ]
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out for %s", path)
            continue

        if result.returncode == 0:
            outputs.append(str(out))
        else:
            logger.warning("ffmpeg error: %s", result.stderr.decode(errors="replace"))

    return outputs


# ═══════════════════════ control panel ═══════════════════════════════

def _default_features() -> Dict[str, bool]:
    return {k: True for k in ALL_FEATURES}


def _build_keyboard(features: Dict[str, bool]) -> InlineKeyboardMarkup:
    def btn(key: str) -> InlineKeyboardButton:
        emoji, name = FEATURE_LABELS[key]
        mark = "✅" if features.get(key, True) else "⬜"
        return InlineKeyboardButton(
            f"{mark} {emoji} {name}", callback_data=f"toggle:{key}"
        )

    return InlineKeyboardMarkup([
        [btn(F_NOISE), btn(F_BG)],
        [btn(F_COLOR), btn(F_FONT)],
        [btn(F_CAPTION)],
        [
            InlineKeyboardButton("↩️ 全选", callback_data="select_all"),
            InlineKeyboardButton("🗑 清空", callback_data="select_none"),
        ],
        [InlineKeyboardButton("🚀 开始生成", callback_data="generate")],
    ])


def _panel_text(media_type: str) -> str:
    label = "图片" if media_type == "photo" else "视频"
    count = IMAGE_VARIANTS if media_type == "photo" else VIDEO_VARIANTS
    return (
        f"✅ 已收到{label}，将生成 *{count}* 个变体。\n\n"
        "请选择裂变功能，然后点击 🚀 开始生成："
    )


# ═══════════════════════ telegram handlers ════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *PHSUGAR 素材裂变Bot*\n\n"
        "发送一张 *图片* 或 *视频* 即可进入控制面板。\n\n"
        "支持的裂变功能：\n"
        "🔀 噪点 — 随机像素噪点\n"
        "🖼 换背景 — 替换图片背景\n"
        "🎨 换颜色 — 随机色彩调整\n"
        "✏️ 换字体 — 品牌水印叠加\n"
        "💬 换字幕 — 底部字幕栏\n\n"
        "所有功能可自由组合，支持批量生成。",
        parse_mode="Markdown",
    )


async def _show_panel(
    update: Update, context: ContextTypes.DEFAULT_TYPE, media_type: str
) -> None:
    features = _default_features()
    context.user_data["features"] = features
    context.user_data["media_type"] = media_type

    msg = await update.message.reply_text(
        _panel_text(media_type),
        reply_markup=_build_keyboard(features),
        parse_mode="Markdown",
    )
    context.user_data["panel_message_id"] = msg.message_id


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file = await update.message.photo[-1].get_file()
    path = WORK_DIR / f"{uuid.uuid4().hex}.jpg"
    await file.download_to_drive(str(path))
    context.user_data["pending_path"] = str(path)
    await _show_panel(update, context, "photo")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file = await update.message.video.get_file()
    path = WORK_DIR / f"{uuid.uuid4().hex}.mp4"
    await file.download_to_drive(str(path))
    context.user_data["pending_path"] = str(path)
    await _show_panel(update, context, "video")


async def handle_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()

    data: str = query.data
    features: Dict[str, bool] = context.user_data.get("features", _default_features())
    media_type: str = context.user_data.get("media_type", "photo")

    # ── toggle a single feature ───────────────────────────────────────
    if data.startswith("toggle:"):
        key = data.split(":", 1)[1]
        if key in ALL_FEATURES:
            features[key] = not features.get(key, True)
            context.user_data["features"] = features
            await query.edit_message_text(
                _panel_text(media_type),
                reply_markup=_build_keyboard(features),
                parse_mode="Markdown",
            )
        return

    # ── select-all / clear ────────────────────────────────────────────
    if data in ("select_all", "select_none"):
        features = {k: data == "select_all" for k in ALL_FEATURES}
        context.user_data["features"] = features
        await query.edit_message_text(
            _panel_text(media_type),
            reply_markup=_build_keyboard(features),
            parse_mode="Markdown",
        )
        return

    # ── generate ──────────────────────────────────────────────────────
    if data == "generate":
        pending = context.user_data.get("pending_path", "")
        if not pending or not os.path.exists(pending):
            await query.edit_message_text(
                "❌ 没有待处理的素材，请重新发送图片或视频。"
            )
            return

        await query.edit_message_text("⏳ 正在生成变体，请稍候…")

        outputs: List[str] = []
        try:
            if media_type == "photo":
                outputs = generate_image_variants(pending, features)
                for img_path in outputs:
                    with open(img_path, "rb") as f:
                        await query.message.reply_photo(photo=f)
            else:
                outputs = generate_video_variants(pending, features)
                for vid_path in outputs:
                    with open(vid_path, "rb") as f:
                        await query.message.reply_video(video=f)

            await query.message.reply_text(
                f"✅ 已生成 {len(outputs)} 个变体！\n发送新素材可继续裂变。"
            )
        except Exception:
            logger.exception("Generation error")
            await query.message.reply_text("❌ 生成失败，请检查素材后重试。")
        finally:
            # Clean up generated variants and the original upload
            for p in outputs:
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.remove(pending)
            except OSError:
                pass
            context.user_data.pop("pending_path", None)


# ═════════════════════════════ main ══════════════════════════════════

def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("TG_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Bot started…")
    app.run_polling()


if __name__ == "__main__":
    main()
