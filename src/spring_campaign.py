from __future__ import annotations
import os.path

from fastapi import APIRouter
from pydantic import BaseModel

import asyncio
import base64
from pathlib import Path
from typing import Iterable, List
from xml.etree import ElementTree as ET

import httpx
from PIL import Image, ImageDraw, ImageFont


class CardResponse(BaseModel):
    card_content: str


RES_DIR = Path("/tmp/2026-spring")
FONT_URL = "https://solulu-app.tos-cn-shanghai.volces.com/campaign/2026-spring/PingFangSC-Regular.ttf"
FONT_PATH = RES_DIR / "PingFangSC-Regular.ttc"
AVATAR_RELATIVE_URL = "https://solulu-app.tos-cn-shanghai.volces.com/campaign/2026-spring/avatar_relative.svg"
AVATAR_RELATIVE_PATH = RES_DIR / "avatar_relative.svg"
AVATAR_ME_URL = "https://solulu-app.tos-cn-shanghai.volces.com/campaign/2026-spring/avatar_me.svg"
AVATAR_ME_PATH = RES_DIR / "avatar_me.svg"
TEMPLATE_URL = "https://solulu-app.tos-cn-shanghai.volces.com/campaign/2026-spring/spring_template.svg"
TEMPLATE_PATH = RES_DIR / "spring_template.svg"
router = APIRouter()

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Register namespaces so we do not lose them when writing the SVG back.
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


class BubbleStyle:
    """Layout values that control how each chat bubble is drawn."""

    def __init__(
            self,
            *,
            x: int,
            start_y: int,
            max_width: int,
            avatar_gap: int,
            padding_x: int,
            padding_y: int,
            radius: int,
            line_spacing: int,
            gap: int,
            bubble_fill: str,
            summary_fill: str,
            text_fill: str,
            font_size: int,
            summary_bottom_margin: int,
            summary_max_width: int | None = None,
            summary_x: int | None = None,
    ) -> None:
        self.x = x
        self.start_y = start_y
        self.max_width = max_width
        self.avatar_gap = avatar_gap
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.radius = radius
        self.line_spacing = line_spacing
        self.gap = gap
        self.bubble_fill = bubble_fill
        self.summary_fill = summary_fill
        self.text_fill = text_fill
        self.font_size = font_size
        self.summary_bottom_margin = summary_bottom_margin
        self.summary_max_width = summary_max_width
        self.summary_x = summary_x if summary_x is not None else x


STYLE = BubbleStyle(
    x=22,
    start_y=110,
    max_width=295,
    avatar_gap=8,
    padding_x=12,
    padding_y=10,
    radius=12,
    line_spacing=6,
    gap=12,
    bubble_fill="#FFFFFF",
    summary_fill="#FFE6BF",
    text_fill="#352E26",
    font_size=15,
    summary_bottom_margin=100,
    summary_max_width=260,
    summary_x=170,
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
                "dominant-baseline": "text-before-edge",
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
) -> tuple[List[str], int, int, int]:
    """Calculate lines, line height, bubble width and height for a given text."""
    max_width = max_width or style.max_width
    text_max_width = max_width - style.padding_x * 2
    lines = _wrap_text(draw, text, font, text_max_width)

    # Measure vertical metrics once; line height comes from a single glyph.
    line_height = _measure_text(draw, "啊", font)[1]
    text_block_height = (line_height * len(lines)) + (style.line_spacing * (len(lines) - 1 if len(lines) > 1 else 0))

    bubble_height = text_block_height + style.padding_y * 2
    text_widths = [_measure_text(draw, line, font)[0] for line in lines]
    bubble_width = min(max_width, max(text_widths) + style.padding_x * 2 if text_widths else max_width)

    return lines, line_height, bubble_width, bubble_height


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
        bubble_width: int,
        bubble_height: int,
        y: int,
        style: BubbleStyle,
        fill: str,
        x_override: int | None = None,
) -> None:
    """Render a single bubble into the parent group."""
    x = x_override if x_override is not None else style.x
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
    text_y = y + style.padding_y
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


def render_svg(relatives_message: str, my_message: str) -> str:
    """Render chat bubbles into the base Spring template and return SVG text."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = ImageFont.truetype(str(FONT_PATH), STYLE.font_size)

    template = ET.parse(TEMPLATE_PATH)
    root = template.getroot()
    canvas_width = _parse_svg_length(root.get("width"))
    view_box = root.get("viewBox")
    if canvas_width is None and view_box:
        parts = view_box.split()
        if len(parts) == 4:
            canvas_width = _parse_svg_length(parts[2])
    if canvas_width is None:
        raise ValueError("SVG template is missing width or viewBox width.")
    canvas_width_int = int(canvas_width)

    chat_group = ET.Element(_svg_tag("g"), {"id": "campaign-chat"})
    root.append(chat_group)

    relatives_avatar_href, relatives_avatar_width, relatives_avatar_height = _load_avatar_svg(AVATAR_RELATIVE_PATH)
    my_avatar_href, my_avatar_width, my_avatar_height = _load_avatar_svg(AVATAR_ME_PATH)

    bubble_start_x = STYLE.x + relatives_avatar_width + STYLE.avatar_gap
    base_bubble_max_width = max(1, STYLE.max_width - (bubble_start_x - STYLE.x))
    my_avatar_x = canvas_width_int - STYLE.x - my_avatar_width
    my_bubble_max_width = max(1, min(base_bubble_max_width, my_avatar_x - STYLE.avatar_gap - bubble_start_x))

    current_y = STYLE.start_y

    # Render the first bubble.
    relatives_lines, relatives_line_height, relatives_bubble_width, relatives_bubble_height = _bubble_layout(
        draw,
        avatar="",
        text=relatives_message,
        font=font,
        style=STYLE,
        max_width=base_bubble_max_width,
    )
    _render_avatar(
        chat_group,
        href=relatives_avatar_href,
        x=STYLE.x,
        y=current_y,
        width=relatives_avatar_width,
        height=relatives_avatar_height,
    )
    _render_bubble(
        chat_group,
        lines=relatives_lines,
        line_height=relatives_line_height,
        bubble_width=relatives_bubble_width,
        bubble_height=relatives_bubble_height,
        y=current_y,
        style=STYLE,
        fill=STYLE.bubble_fill,
        x_override=bubble_start_x,
    )
    current_y += max(relatives_bubble_height, relatives_avatar_height) + STYLE.gap

    # Render the second bubble.
    my_lines, my_line_height, my_bubble_width, my_bubble_height = _bubble_layout(
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

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


async def download_res(res_path: Path, res_url: str):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(res_url)
        response.raise_for_status()
        data = response.content
    await asyncio.to_thread(res_path.write_bytes, data)


async def ensure_resources():
    for res in zip([FONT_PATH, TEMPLATE_PATH, AVATAR_ME_PATH, AVATAR_RELATIVE_PATH],
                   [FONT_URL, TEMPLATE_URL, AVATAR_ME_URL, AVATAR_RELATIVE_URL]):
        if not os.path.exists(res[0]):
            await download_res(res[0], res[1])


@router.post("/api/campaign/2026-spring/generate-card", response_model=CardResponse)
async def generate_spring_card(relatives_message: str, my_message: str) -> CardResponse:
    await ensure_resources()
    output_svg = render_svg(relatives_message, my_message)
    return CardResponse(card_content=output_svg)

if __name__ == '__main__':
    result = render_svg("今年过节不收礼", "收礼只收脑白金，你是不是指望我回你这句呢")
    with open("output.svg", "w", encoding="utf-8") as f:
        f.write(result)
