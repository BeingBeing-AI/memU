from __future__ import annotations
import os.path
import shutil

from fastapi import APIRouter
from pydantic import BaseModel

import asyncio
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List
from xml.etree import ElementTree as ET

import httpx
from PIL import Image, ImageDraw, ImageFont
from starlette.responses import JSONResponse


class CardRequest(BaseModel):
    relatives_message: str
    my_message: str

class CardResponse(BaseModel):
    card_content: str


RES_DIR = Path("2026-spring")
FONT_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/PingFangSC-Regular.ttf"
FONT_PATH = RES_DIR / "PingFangSC-Regular.ttc"
AVATAR_RELATIVE_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/avatar_relative.svg"
AVATAR_RELATIVE_PATH = RES_DIR / "avatar_relative.svg"
AVATAR_ME_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/avatar_me.svg"
AVATAR_ME_PATH = RES_DIR / "avatar_me.svg"
BACKGROUND_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/background.svg"
BACKGROUND_PATH = RES_DIR / "background.svg"
BOTTOM_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/bottom.svg"
BOTTOM_PATH = RES_DIR / "bottom.svg"

DOT_1_LAYER_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/dot_short.svg"
DOT_1_LAYER_PATH = RES_DIR / "dot_1_layer.svg"
DOT_2_LAYER_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/dot_short.svg"
DOT_2_LAYER_PATH = RES_DIR / "dot_2_layer.svg"
DOT_SHORT_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/dot_short.svg"
DOT_SHORT_PATH = RES_DIR / "dot_short.svg"
DOT_LONG_URL = "https://solulu-cdn.starfyally.com/campaign/2026-spring/dot_long.svg"
DOT_LONG_PATH = RES_DIR / "dot_long.svg"
router = APIRouter()

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Register namespaces so we do not lose them when writing the SVG back.
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


@dataclass(frozen=True, slots=True)
class BubbleStyle:
    """Layout values that control how each chat bubble is drawn."""
    left_offset: int
    right_offset: int
    start_y: int
    max_width: int
    avatar_gap: int
    padding_x: int
    padding_y: int
    radius: int
    line_spacing: int
    gap: int
    dot_line_ranges: tuple[tuple[int, int | None, Path], ...]
    outer_height_offset: int
    dot_y_offset: int
    bubble_fill: str
    summary_fill: str
    text_fill: str
    font_size: int


STYLE = BubbleStyle(
    left_offset=13,
    right_offset=8,
    start_y=39,
    max_width=295,
    avatar_gap=8,
    padding_x=12,
    padding_y=10,
    radius=12,
    line_spacing=0,
    gap=18,
    dot_line_ranges=(
        (0, 2, DOT_1_LAYER_PATH),
        (2, 6, DOT_2_LAYER_PATH),
        (6, 10, DOT_SHORT_PATH),
        (10, None, DOT_LONG_PATH),
    ),
    outer_height_offset=130,
    dot_y_offset=10,
    bubble_fill="#FFFFFF",
    summary_fill="#FFE6BF",
    text_fill="#352E26",
    font_size=14,
)


def _svg_tag(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Greedy wrap on a per-character basis so Chinese text breaks correctly."""
    if not text:
        return [""]

    lines: List[str] = []
    current = ""

    for ch in text:
        test = current + ch
        w, _ = _measure_text(draw, test, font)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = ch

    if current:
        lines.append(current)

    return lines or [""]


def _create_text_elements(
        group: ET.Element,
        lines: Iterable[str],
        *,
        x: int,
        y: int,
        line_height: int,
        line_spacing: int,
        font_size: int,
        text_fill: str,
) -> None:
    """Add <text> elements for each line into the given SVG group."""
    for idx, line in enumerate(lines):
        text_y = y + idx * (line_height + line_spacing)
        text_el = ET.Element(
            _svg_tag("text"),
            {
                "x": str(x),
                "y": str(text_y),
                "font-family": "PingFang SC, Noto Sans SC, sans-serif",
                "font-size": str(font_size),
                "fill": text_fill,
            },
        )
        text_el.text = line
        group.append(text_el)


def _bubble_layout(
        draw: ImageDraw.ImageDraw,
        *,
        avatar: str,
        text: str,
        font: ImageFont.FreeTypeFont,
        style: BubbleStyle,
        max_width: int | None = None,
) -> tuple[List[str], int, int, int, int]:
    """Calculate lines, line height, bubble width and height for a given text."""
    max_width = max_width or style.max_width
    text_max_width = max_width - style.padding_x * 2
    lines = _wrap_text(draw, text, font, text_max_width)

    # Use font metrics so SVG baseline positioning matches the bubble height.
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    text_block_height = (line_height * len(lines)) + (style.line_spacing * (len(lines) - 1 if len(lines) > 1 else 0))

    bubble_height = text_block_height + style.padding_y * 2
    text_widths = [_measure_text(draw, line, font)[0] for line in lines]
    bubble_width = min(max_width, max(text_widths) + style.padding_x * 2 if text_widths else max_width)

    return lines, line_height, ascent, bubble_width, bubble_height


def _parse_svg_length(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.endswith("px"):
        cleaned = cleaned[:-2]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_rect(root: ET.Element, rect_id: str) -> ET.Element:
    for rect in root.iter(_svg_tag("rect")):
        if rect.get("id") == rect_id:
            return rect
    raise ValueError(f"SVG template missing rect id={rect_id}")


def _parse_rect_geometry(rect: ET.Element) -> tuple[float, float, float, float]:
    x = _parse_svg_length(rect.get("x")) or 0.0
    y = _parse_svg_length(rect.get("y")) or 0.0
    width = _parse_svg_length(rect.get("width"))
    height = _parse_svg_length(rect.get("height"))
    if width is None or height is None:
        raise ValueError("SVG rect is missing width/height.")
    return x, y, width, height


def _parse_svg_size(root: ET.Element) -> tuple[float | None, float | None]:
    width = _parse_svg_length(root.get("width"))
    height = _parse_svg_length(root.get("height"))
    view_box = root.get("viewBox")
    if (width is None or height is None) and view_box:
        parts = view_box.split()
        if len(parts) == 4:
            width = width or _parse_svg_length(parts[2])
            height = height or _parse_svg_length(parts[3])
    return width, height


def _load_avatar_svg(avatar_path: str | Path) -> tuple[str, int, int]:
    avatar_path = Path(avatar_path)
    avatar_text = avatar_path.read_text(encoding="utf-8")
    avatar_root = ET.fromstring(avatar_text)

    width = _parse_svg_length(avatar_root.get("width"))
    height = _parse_svg_length(avatar_root.get("height"))
    view_box = avatar_root.get("viewBox")
    if (width is None or height is None) and view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            width = width or _parse_svg_length(parts[2])
            height = height or _parse_svg_length(parts[3])

    if width is None or height is None:
        raise ValueError(f"Avatar SVG is missing size info: {avatar_path}")

    encoded = base64.b64encode(avatar_text.encode("utf-8")).decode("ascii")
    data_uri = f"data:image/svg+xml;base64,{encoded}"
    return data_uri, int(width), int(height)


def _render_avatar(parent: ET.Element, *, href: str, x: int, y: int, width: int, height: int) -> None:
    image = ET.Element(
        _svg_tag("image"),
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "href": href,
            f"{{{XLINK_NS}}}href": href,
        },
    )
    parent.append(image)


def _render_bubble(
        parent: ET.Element,
        *,
        lines: List[str],
        line_height: int,
        baseline_offset: int,
        bubble_width: int,
        bubble_height: int,
        y: int,
        style: BubbleStyle,
        fill: str,
        x_override: int | None = None,
) -> None:
    """Render a single bubble into the parent group."""
    x = x_override if x_override is not None else style.left_offset
    rect = ET.Element(
        _svg_tag("rect"),
        {
            "x": str(x),
            "y": str(y),
            "width": str(bubble_width),
            "height": str(bubble_height),
            "rx": str(style.radius),
            "fill": fill,
        },
    )
    parent.append(rect)

    text_x = x + style.padding_x
    text_y = y + style.padding_y + baseline_offset
    _create_text_elements(
        parent,
        lines,
        x=text_x,
        y=text_y,
        line_height=line_height,
        line_spacing=style.line_spacing,
        font_size=style.font_size,
        text_fill=style.text_fill,
    )


def _ensure_height(root: ET.Element, required_height: int) -> None:
    """Grow the SVG canvas if new bubbles overflow the original size."""

    def _parse_number(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    current_height = _parse_number(root.get("height")) or required_height
    if required_height <= current_height:
        return

    root.set("height", str(required_height))
    view_box = root.get("viewBox")
    if view_box:
        parts = view_box.split()
        if len(parts) == 4:
            parts[3] = str(required_height)
            root.set("viewBox", " ".join(parts))

def _select_dot_path(line_count: int, ranges: tuple[tuple[int, int | None, Path], ...]) -> Path:
    for min_lines, max_lines, dot_path in ranges:
        if line_count < min_lines:
            continue
        if max_lines is None or line_count < max_lines:
            return dot_path
    return DOT_LONG_PATH


def render_svg(relatives_message: str, my_message: str) -> str:
    """Render chat bubbles into the base Spring template and return SVG text."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = ImageFont.truetype(str(FONT_PATH), STYLE.font_size)

    template = ET.parse(BACKGROUND_PATH)
    root = template.getroot()
    outer_rect = _find_rect(root, "b_outer")
    inner_rect = _find_rect(root, "b_inner")
    outer_x, _, _, _ = _parse_rect_geometry(outer_rect)
    inner_x, inner_y, inner_width, inner_height = _parse_rect_geometry(inner_rect)

    canvas_width, _ = _parse_svg_size(root)
    if canvas_width is None:
        raise ValueError("SVG template is missing width or viewBox width.")

    chat_group = ET.Element(_svg_tag("g"), {"id": "campaign-chat"})
    root.append(chat_group)

    relatives_avatar_href, relatives_avatar_width, relatives_avatar_height = _load_avatar_svg(AVATAR_RELATIVE_PATH)
    my_avatar_href, my_avatar_width, my_avatar_height = _load_avatar_svg(AVATAR_ME_PATH)

    left_margin = max(0, STYLE.left_offset)
    right_margin = max(0, STYLE.right_offset)
    top_margin = max(0, STYLE.start_y - inner_y)
    chat_left = inner_x + left_margin
    chat_right = inner_x + inner_width - right_margin

    bubble_start_x = int(chat_left + relatives_avatar_width + STYLE.avatar_gap)
    base_bubble_max_width = max(1, STYLE.max_width)
    my_avatar_x = int(chat_right - my_avatar_width)
    bubble_right_limit = my_avatar_x - STYLE.avatar_gap - bubble_start_x
    relatives_bubble_max_width = max(1, min(base_bubble_max_width, bubble_right_limit))
    my_bubble_max_width = max(1, min(base_bubble_max_width, bubble_right_limit))

    current_y = int(inner_y + top_margin)

    # Render the first bubble.
    relatives_lines, relatives_line_height, relatives_baseline_offset, relatives_bubble_width, relatives_bubble_height = _bubble_layout(
        draw,
        avatar="",
        text=relatives_message,
        font=font,
        style=STYLE,
        max_width=relatives_bubble_max_width,
    )
    _render_avatar(
        chat_group,
        href=relatives_avatar_href,
        x=int(chat_left),
        y=current_y,
        width=relatives_avatar_width,
        height=relatives_avatar_height,
    )
    _render_bubble(
        chat_group,
        lines=relatives_lines,
        line_height=relatives_line_height,
        baseline_offset=relatives_baseline_offset,
        bubble_width=relatives_bubble_width,
        bubble_height=relatives_bubble_height,
        y=current_y,
        style=STYLE,
        fill=STYLE.bubble_fill,
        x_override=bubble_start_x,
    )
    current_y += relatives_bubble_height + STYLE.gap

    # Render the second bubble.
    my_lines, my_line_height, my_baseline_offset, my_bubble_width, my_bubble_height = _bubble_layout(
        draw,
        avatar="",
        text=my_message,
        font=font,
        style=STYLE,
        max_width=my_bubble_max_width,
    )
    _render_avatar(
        chat_group,
        href=my_avatar_href,
        x=my_avatar_x,
        y=current_y,
        width=my_avatar_width,
        height=my_avatar_height,
    )
    _render_bubble(
        chat_group,
        lines=my_lines,
        line_height=my_line_height,
        baseline_offset=my_baseline_offset,
        bubble_width=my_bubble_width,
        bubble_height=my_bubble_height,
        y=current_y,
        style=STYLE,
        fill="#FBC966",
        x_override=(
            my_avatar_x - STYLE.avatar_gap - my_bubble_width
            if len(my_lines) == 1
            else bubble_start_x
        ),
    )
    current_y += max(my_bubble_height, my_avatar_height) + STYLE.gap

    required_inner_height = current_y - inner_y
    inner_rect.set("height", str(int(required_inner_height)))

    required_outer_height = int(required_inner_height + STYLE.outer_height_offset)
    outer_rect.set("height", str(required_outer_height))

    total_lines = max(1, len(relatives_lines) + len(my_lines))
    dot_path = _select_dot_path(total_lines, STYLE.dot_line_ranges)
    dot_tree = ET.parse(dot_path)
    dot_root = dot_tree.getroot()
    dot_width, _ = _parse_svg_size(dot_root)
    dot_translate_x = 0
    if dot_width is not None and canvas_width is not None:
        dot_translate_x = int((canvas_width - dot_width) / 2)
    dot_group = ET.Element(
        _svg_tag("g"),
        {"id": "campaign-dot", "transform": f"translate({dot_translate_x} {-STYLE.dot_y_offset})"},
    )
    for child in list(dot_root):
        dot_group.append(child)
    root.insert(1, dot_group)

    bottom_tree = ET.parse(BOTTOM_PATH)
    bottom_root = bottom_tree.getroot()
    bottom_width, bottom_height = _parse_svg_size(bottom_root)
    if bottom_height is None:
        raise ValueError("Bottom SVG is missing height or viewBox height.")

    translate_x = int(outer_x)

    translate_y = int(required_outer_height - bottom_height)
    bottom_group = ET.Element(
        _svg_tag("g"),
        {"id": "campaign-bottom", "transform": f"translate({translate_x} {translate_y})"},
    )
    for child in list(bottom_root):
        bottom_group.append(child)
    root.append(bottom_group)

    _ensure_height(root, required_outer_height)

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


async def download_res(res_path: Path, res_url: str):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(res_url)
        response.raise_for_status()
        data = response.content
    await asyncio.to_thread(res_path.write_bytes, data)


async def ensure_resources():
    resources = [
        (FONT_PATH, FONT_URL),
        (BACKGROUND_PATH, BACKGROUND_URL),
        (DOT_1_LAYER_PATH, DOT_1_LAYER_URL),
        (DOT_2_LAYER_PATH, DOT_2_LAYER_URL),
        (DOT_SHORT_PATH, DOT_SHORT_URL),
        (DOT_LONG_PATH, DOT_LONG_URL),
        (BOTTOM_PATH, BOTTOM_URL),
        (AVATAR_ME_PATH, AVATAR_ME_URL),
        (AVATAR_RELATIVE_PATH, AVATAR_RELATIVE_URL),
    ]
    for res_path, res_url in resources:
        if not os.path.exists(res_path):
            await download_res(res_path, res_url)


@router.post("/api/campaign/2026-spring/generate-card", response_model=CardResponse)
async def generate_spring_card(request: CardRequest):
    await ensure_resources()
    output_svg = render_svg(request.relatives_message, request.my_message)
    return JSONResponse(content={"content": output_svg})

@router.post("/api/campaign/2026-spring/reload-resource")
async def reload_resource():
    shutil.rmtree(str(RES_DIR))
    await ensure_resources()
    return JSONResponse(content={"content": "success"})


async def test():
    await ensure_resources()
    result = render_svg("今年过节不收礼", "收礼只收脑白金")
    # result = render_svg("今年过节不收礼", "收礼只收脑白金，你是不是指望我回你这句呢")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # result = render_svg("今年过节不收礼哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "收礼只收脑白金，你是不是指望我回你这句呢，哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    with open("output.svg", "w", encoding="utf-8") as f:
        f.write(result)


if __name__ == '__main__':
    asyncio.run(test())
