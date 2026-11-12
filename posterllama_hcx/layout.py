"""Parse coordinates and draw boxes without executing generated HTML."""
import math
from dataclasses import dataclass
from html.parser import HTMLParser

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Box:
    category: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Layout:
    width: float
    height: float
    boxes: tuple[Box, ...]


class _LayoutParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.boxes = []
        self.inside_svg = False
        self.closed_svg = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "svg":
            if self.canvas is not None:
                raise ValueError("Expected exactly one SVG canvas")
            self.canvas = (self.number(attrs, "width"), self.number(attrs, "height"))
            self.inside_svg = True
        elif tag == "rect":
            if not self.inside_svg:
                raise ValueError("Rectangle must be inside SVG")
            category = attrs.get("data-category", attrs.get("category", "text"))
            self.boxes.append(Box(category, *(self.number(attrs, k) for k in ("x", "y", "width", "height"))))

    def handle_endtag(self, tag):
        if tag == "svg":
            self.inside_svg = False
            self.closed_svg = True

    @staticmethod
    def number(attrs, key):
        try:
            value = float(attrs[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or nonnumeric {key} in generated layout") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {key}")
        return value


def parse_layout(html):
    parser = _LayoutParser()
    parser.feed(html)
    if not parser.canvas or not parser.closed_svg or not parser.boxes:
        raise ValueError("Expected a complete <svg> with at least one <rect>")
    width, height = parser.canvas
    if width <= 0 or height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    for box in parser.boxes:
        if box.x < 0 or box.y < 0 or box.width <= 0 or box.height <= 0:
            raise ValueError("Invalid box coordinates/dimensions")
        if box.x + box.width > width + 1e-4 or box.y + box.height > height + 1e-4:
            raise ValueError("Box extends beyond the canvas")
    return Layout(width, height, tuple(parser.boxes))


def render_layout(image, html, output_path=None):
    """Overlay layout boxes on the original image; this is not a typography renderer."""
    layout = parse_layout(html)
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    scale_x, scale_y = canvas.width / layout.width, canvas.height / layout.height
    colors = {"text": "#0c94cf", "logo": "#e45525", "underlay": "#599a45", "underlay_bg": "#599a45"}
    for box in layout.boxes:
        bounds = (box.x * scale_x, box.y * scale_y,
                  (box.x + box.width) * scale_x, (box.y + box.height) * scale_y)
        color = colors.get(box.category, "#b246b0")
        draw.rectangle(bounds, outline=color, width=3)
        draw.text((bounds[0] + 3, bounds[1] + 3), box.category, fill=color)
    if output_path:
        canvas.save(output_path)
    return canvas
