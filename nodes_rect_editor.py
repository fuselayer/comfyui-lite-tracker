import json
import os
import time
from typing import Tuple, List, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw

from folder_paths import get_temp_directory  


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _grid_in_rect_local(w: int, h: int,
                        density_per_100px: float,
                        margin_px: int,
                        jitter_px: float,
                        seed: int) -> np.ndarray:
    """
    Generate grid points in local coordinate space, centered at origin.
    Returns points in range [-w/2, w/2] x [-h/2, h/2].
    """
    w_eff = max(1, w - 2 * margin_px)
    h_eff = max(1, h - 2 * margin_px)

    cols = max(1, int(round((w_eff / 100.0) * float(density_per_100px))))
    rows = max(1, int(round((h_eff / 100.0) * float(density_per_100px))))

    if cols == 1:
        xs = np.array([0.0], dtype=np.float32)
    else:
        xs = np.linspace(-w_eff / 2.0, w_eff / 2.0, cols, dtype=np.float32)
    
    if rows == 1:
        ys = np.array([0.0], dtype=np.float32)
    else:
        ys = np.linspace(-h_eff / 2.0, h_eff / 2.0, rows, dtype=np.float32)

    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)

    if jitter_px and jitter_px > 0:
        rng = np.random.RandomState(seed)
        jitter = (rng.rand(*pts.shape).astype(np.float32) * 2.0 - 1.0) * float(jitter_px)
        pts = pts + jitter

    return pts


def _transform_points(points: np.ndarray, 
                     cx: float, cy: float, 
                     rotation: float) -> np.ndarray:
    """
    Rotate points around origin, then translate to (cx, cy).
    points: [N, 2] in local space
    rotation: angle in radians
    """
    if rotation != 0:
        cos_a = np.cos(rotation)
        sin_a = np.sin(rotation)
        x_rot = points[:, 0] * cos_a - points[:, 1] * sin_a
        y_rot = points[:, 0] * sin_a + points[:, 1] * cos_a
        points = np.stack([x_rot, y_rot], axis=-1)
    
    points[:, 0] += cx
    points[:, 1] += cy
    return points


def _draw_overlay_multi(img: torch.Tensor,
                       rectangles: List[Dict],
                       density_per_100px: float,
                       margin_px: int,
                       jitter_px: float,
                       seed: int,
                       color_rect=(0, 255, 255),
                       color_pts=(0, 255, 0)) -> torch.Tensor:
    """
    Draw multiple rotated rectangles + points on IMAGE [T,H,W,C] float 0..1.
    Returns IMAGE [1,H,W,C] float 0..1.
    """
    if img.ndim != 4:
        raise ValueError("image must be IMAGE [T,H,W,C]")

    base = (img[0].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    pil = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(pil)

    for rect_data in rectangles:
        x = float(rect_data.get("x", 0))
        y = float(rect_data.get("y", 0))
        w = float(rect_data.get("w", 100))
        h = float(rect_data.get("h", 100))
        rotation = float(rect_data.get("rotation", 0))

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Draw rotated rectangle outline
        corners = np.array([
            [-w/2, -h/2],
            [ w/2, -h/2],
            [ w/2,  h/2],
            [-w/2,  h/2]
        ], dtype=np.float32)

        if rotation != 0:
            cos_a = np.cos(rotation)
            sin_a = np.sin(rotation)
            x_rot = corners[:, 0] * cos_a - corners[:, 1] * sin_a
            y_rot = corners[:, 0] * sin_a + corners[:, 1] * cos_a
            corners = np.stack([x_rot, y_rot], axis=-1)
        
        corners[:, 0] += cx
        corners[:, 1] += cy

        for i in range(4):
            p1 = (int(corners[i, 0]), int(corners[i, 1]))
            p2 = (int(corners[(i+1) % 4, 0]), int(corners[(i+1) % 4, 1]))
            draw.line([p1, p2], fill=tuple(color_rect), width=2)

        # Generate and draw points for this rectangle
        local_pts = _grid_in_rect_local(int(w), int(h), density_per_100px, margin_px, jitter_px, seed)
        world_pts = _transform_points(local_pts, cx, cy, rotation)

        r = 2
        for p in world_pts:
            px = int(round(p[0]))
            py = int(round(p[1]))
            if 0 <= px < pil.width and 0 <= py < pil.height:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=tuple(color_pts))

    out = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)
    return out


def _save_png(image_bhwc_float: torch.Tensor, prefix="lt_grid_editor") -> str:
    """
    Save IMAGE [1,H,W,C] float 0..1 to temp as PNG and return filename (not full path).
    """
    save_dir = get_temp_directory()
    os.makedirs(save_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{ts}.png"
    path = os.path.join(save_dir, filename)
    im = (image_bhwc_float[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(im, mode="RGB").save(path, format="PNG")
    return filename


class RectEditor:
    """
    LiteTracker: Grid Editor (Multi-Rectangle with Rotation)
    - Click 'Open Grid Editor' to open modal
    - Click-drag to draw rectangles (adds to array)
    - Drag rotation handle (small circle at top) to rotate each rectangle
    - Right-click on rectangle to delete it
    - Undo button removes last rectangle
    - Clear button removes all rectangles
    - Generates grid points for all rectangles combined
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "density_per_100px": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "rect_json": ("STRING", {"default": ""}),
                "x": ("INT", {"default": 100, "min": 0, "max": 100000}),
                "y": ("INT", {"default": 100, "min": 0, "max": 100000}),
                "w": ("INT", {"default": 400, "min": 1, "max": 100000}),
                "h": ("INT", {"default": 400, "min": 1, "max": 100000}),
                "margin_px": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "random_jitter_px": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("TENSOR", "IMAGE", "BBOX", "STRING", "STRING")
    RETURN_NAMES = ("points", "preview_png", "bbox", "rect_json", "bg_base64")
    FUNCTION = "run"
    CATEGORY = "LiteTracker"

    def run(self,
            image: torch.Tensor,
            density_per_100px: float,
            rect_json: str = "",
            x: int = 100, y: int = 100, w: int = 400, h: int = 400,
            margin_px: int = 0,
            random_jitter_px: float = 0.0,
            seed: int = 0):

        if image.ndim != 4:
            raise ValueError("image must be IMAGE [T,H,W,C]")

        H = int(image.shape[1])
        W = int(image.shape[2])

        # Parse rectangles array from rect_json
        rectangles = []
        if rect_json and rect_json.strip():
            try:
                parsed = json.loads(rect_json)
                if isinstance(parsed, list):
                    rectangles = parsed
                elif isinstance(parsed, dict):
                    # Old single-rectangle format
                    rectangles = [parsed]
            except Exception:
                pass
        
        # Fallback: use manual x/y/w/h inputs
        if not rectangles:
            rectangles = [{"x": x, "y": y, "w": w, "h": h, "rotation": 0}]

        # Clamp margin/jitter
        max_margin = max(0, min(W, H) // 2)
        max_jitter = max(1.0, min(float(min(W, H)) / 4.0, 1000.0))
        margin_px = int(max(0, min(int(margin_px), max_margin)))
        random_jitter_px = float(max(0.0, min(float(random_jitter_px), max_jitter)))

        # Generate points for all rectangles
        all_points = []
        for rect_data in rectangles:
            rx = float(rect_data.get("x", 100))
            ry = float(rect_data.get("y", 100))
            rw = float(rect_data.get("w", 400))
            rh = float(rect_data.get("h", 400))
            rotation = float(rect_data.get("rotation", 0))

            # Clamp rectangle to image bounds (approximately)
            rx = max(0, min(rx, W - 1))
            ry = max(0, min(ry, H - 1))
            rw = max(1, min(rw, W))
            rh = max(1, min(rh, H))

            cx = rx + rw / 2.0
            cy = ry + rh / 2.0

            # Generate points in local space
            local_pts = _grid_in_rect_local(
                int(rw), int(rh),
                density_per_100px,
                margin_px,
                random_jitter_px,
                seed
            )

            # Transform to world space
            world_pts = _transform_points(local_pts, cx, cy, rotation)
            all_points.append(world_pts)

        # Combine all points
        if all_points:
            combined_points = np.concatenate(all_points, axis=0)
        else:
            combined_points = np.zeros((0, 2), dtype=np.float32)

        points_tensor = torch.from_numpy(combined_points.astype(np.float32))

        # Build overlay image
        overlay = _draw_overlay_multi(
            image, rectangles,
            density_per_100px, margin_px, random_jitter_px, seed
        )

        # Compute bounding box of all points
        if len(combined_points) > 0:
            bbox_out = {
                "x": int(np.min(combined_points[:, 0])),
                "y": int(np.min(combined_points[:, 1])),
                "w": int(np.max(combined_points[:, 0]) - np.min(combined_points[:, 0])),
                "h": int(np.max(combined_points[:, 1]) - np.min(combined_points[:, 1]))
            }
        else:
            bbox_out = {"x": 0, "y": 0, "w": 0, "h": 0}

        rect_json_out = json.dumps(rectangles)

        # Encode INPUT image as base64 (not overlay)
        import base64
        import io
        buffer = io.BytesIO()
        im = (image[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # Changed: image instead of overlay
        Image.fromarray(im, mode="RGB").save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "ui": {"bg_base64": [base64_str]}, 
            "result": (points_tensor, overlay, bbox_out, rect_json_out, base64_str)
        }


NODE_CLASS_MAPPINGS = {
    "RectEditor": RectEditor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RectEditor": "LiteTracker: Grid Editor",
}