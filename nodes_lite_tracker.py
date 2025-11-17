import os
import hashlib
from dataclasses import dataclass
from typing import Optional, List
import time

import torch
import numpy as np
import requests
import cv2

# Make the plugin root importable so `import src.*` resolves to our vendored package
import sys
from pathlib import Path

# near the top of the file, after other imports
try:
    from folder_paths import get_temp_directory, get_output_directory
except Exception:
    def get_temp_directory():
        return os.path.join(os.path.dirname(__file__), "results")
    def get_output_directory():
        return os.path.join(os.path.dirname(__file__), "results")

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from src.lite_tracker import LiteTracker
from src.model_utils import get_points_on_a_grid

# Optional import of repo visualizer (lazy)
_VIZ_IMPORT_OK = True
try:
    from src.visualizer import Visualizer
except Exception as _e:
    _VIZ_IMPORT_OK = False
    _VIZ_IMPORT_ERR = _e

DEFAULT_HF_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"

def _models_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models", "lite_tracker")
    return os.path.abspath(base)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _download(url: str, dest_path: str):
    _ensure_dir(os.path.dirname(dest_path))
    import requests
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest_path)

def _pick_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _pick_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32

def _to_uint8_bhwc(img: torch.Tensor) -> torch.Tensor:
    return (img.clamp(0, 1) * 255.0).round().to(torch.uint8)

def _draw_points(batch_bhwc_uint8: torch.Tensor, coords_list: List[np.ndarray], vis_list: Optional[List[np.ndarray]] = None, radius=3, color=(0, 255, 0)) -> torch.Tensor:
    import cv2
    b, h, w, c = batch_bhwc_uint8.shape
    out = batch_bhwc_uint8.clone().cpu().numpy()
    for t in range(b):
        pts = coords_list[t]
        vis = None
        if vis_list is not None and len(vis_list) > t:
            vis = vis_list[t]
        for i, (x, y) in enumerate(pts):
            if vis is not None and not bool(vis[i]):
                continue
            cv2.circle(out[t], (int(round(x)), int(round(y))), radius, color, -1)
    out = torch.from_numpy(out).to(torch.float32) / 255.0
    return out

@dataclass
class LiteTrackerBundle:
    model: LiteTracker
    device: str
    dtype: torch.dtype
    weights_path: str
    queries: Optional[torch.Tensor] = None
    is_initialized: bool = False
    last_coords: Optional[torch.Tensor] = None
    last_vis: Optional[torch.Tensor] = None

class LiteTracker_LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights_source": (["huggingface_scaled_online", "url", "local_path"], {"default": "huggingface_scaled_online"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "url": ("STRING", {"default": DEFAULT_HF_URL}),
                "local_path": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": _models_dir()}),

                # temporal buffer size (safe to adjust)
                "window_len": ("INT", {"default": 16, "min": 4, "max": 64}),

                # per-frame refinement iterations (safe to adjust)
                "iters": ("INT", {"default": 1, "min": 1, "max": 8}),

                # the checkpoint decides this; we will auto-detect/override if needed
                "linear_layer_for_vis_conf": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LITETRACKER",)
    RETURN_NAMES = ("tracker",)
    FUNCTION = "load"
    CATEGORY = "LiteTracker"

    def load(
        self,
        weights_source,
        device,
        url=DEFAULT_HF_URL,
        local_path="",
        cache_dir=_models_dir(),
        window_len=16,
        iters=1,
        linear_layer_for_vis_conf=True,
    ):
        device = _pick_device(device)
        dtype = _pick_dtype(device)

        # Resolve weights path
        if weights_source == "local_path":
            if not local_path:
                raise ValueError("local_path is empty.")
            weights_path = os.path.expanduser(local_path)
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"weights file not found: {weights_path}")
        else:
            _ensure_dir(cache_dir)
            filename = "scaled_online.pth" if weights_source == "huggingface_scaled_online" else os.path.basename((url.split("?")[0] or "weights.pth"))
            weights_path = os.path.join(cache_dir, filename)
            if not os.path.isfile(weights_path):
                _download(DEFAULT_HF_URL if weights_source == "huggingface_scaled_online" else url, weights_path)

        # Inspect weights
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint did not contain a state_dict.")

        # Fixed architecture knobs for this checkpoint family
        corr_levels = 4        # REQUIRED by checkpoint (input_dim contract)
        stride = 4             # REQUIRED by encoder/scaling
        corr_radius = 3        # REQUIRED by corr_mlp dimension

        # Auto-detect num_virtual_tracks and vis head
        nvt = 64
        vt = sd.get("updateformer.virual_tracks", None)
        if isinstance(vt, torch.Tensor) and vt.ndim == 4:
            nvt = vt.shape[1]

        has_vis_head = any(k.startswith("updateformer.vis_conf_head.") for k in sd.keys())
        if has_vis_head != bool(linear_layer_for_vis_conf):
            print(f"[LiteTracker] Adjusting linear_layer_for_vis_conf to {has_vis_head} based on weights.")
            linear_layer_for_vis_conf = has_vis_head

        # Hard-lock model resolution (W,H) = (512,384)
        fixed_model_resolution_w = 512
        fixed_model_resolution_h = 384
        print("[LiteTracker] model_resolution is hard-locked to 512x384.")

        # Build & load
        model = LiteTracker(
            window_len=window_len,
            stride=stride,
            corr_radius=corr_radius,
            corr_levels=corr_levels,
            num_virtual_tracks=nvt,
            model_resolution=(fixed_model_resolution_h, fixed_model_resolution_w),
            linear_layer_for_vis_conf=linear_layer_for_vis_conf,
            iters=iters,
        )
        model.load_state_dict(sd, strict=True)
        model.to(device=device).eval()
        model.init_video_online_processing()

        bundle = LiteTrackerBundle(
            model=model,
            device=device,
            dtype=dtype,
            weights_path=weights_path,
            queries=None,
            is_initialized=False,
        )
        return (bundle,)

class LiteTracker_Track:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracker": ("LITETRACKER",),
                "image": ("IMAGE",),
            },
            "optional": {
                # core tracking
                "reset": ("BOOLEAN", {"default": False}),
                "use_grid": ("BOOLEAN", {"default": True}),
                "grid_size": ("INT", {"default": 10, "min": 1, "max": 256}),
                "grid_query_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "points_text": ("STRING", {"default": ""}),
                "points_tensor": ("TENSOR", {}),   # optional [N,2] or [1,N,2]
                "segm_mask": ("IMAGE", {}),        # optional IMAGE [T,H,W,C] (single frame ok)

                "batch_is_sequence": ("BOOLEAN", {"default": True}),
                "preview_mode": (["simple", "visualizer", "none"], {"default": "visualizer"}),

                # visualizer options (no file saving unless vis_also_save_file=True)
                "vis_also_save_file": ("BOOLEAN", {"default": False}),
                "vis_save_dir": ("STRING", {"default": os.path.join(os.path.dirname(__file__), "results")}),
                "vis_filename": ("STRING", {"default": ""}),
                "vis_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "vis_mode": (["rainbow", "cool", "optical_flow"], {"default": "rainbow"}),
                "vis_linewidth": ("INT", {"default": 2, "min": 1, "max": 12}),
                "vis_show_first_frame": ("INT", {"default": 0, "min": 0, "max": 100}),
                "vis_tracks_leave_trace": ("INT", {"default": 0, "min": -1, "max": 200}),
                "vis_pad_value": ("INT", {"default": 0, "min": 0, "max": 64}),
                "vis_grayscale": ("BOOLEAN", {"default": False}),
                "vis_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "vis_query_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "vis_compensate_for_camera_motion": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LITETRACKER", "TENSOR", "TENSOR", "IMAGE", "STRING",)
    RETURN_NAMES = ("tracker", "coords", "visibility", "preview", "video_path")
    FUNCTION = "track"
    CATEGORY = "LiteTracker"

    def _parse_points(self, s: str) -> Optional[np.ndarray]:
        if not s or not s.strip():
            return None
        pairs = []
        for chunk in s.replace("\n", ";").split(";"):
            chunk = chunk.strip()
            if not chunk or "," not in chunk:
                continue
            xs, ys = chunk.split(",", 1)
            try:
                pairs.append([float(xs.strip()), float(ys.strip())])
            except:
                pass
        if not pairs:
            return None
        return np.asarray(pairs, dtype=np.float32)

    def _build_queries(
        self,
        bundle: LiteTrackerBundle,
        img_uint8_bhwc: torch.Tensor,
        use_grid: bool,
        grid_size: int,
        grid_query_frame: int,
        points_text: str,
        points_tensor: Optional[torch.Tensor] = None,
    ):
        # incoming frame size
        h = int(img_uint8_bhwc.shape[1])
        w = int(img_uint8_bhwc.shape[2])

        # 1) prefer explicit tensor if provided
        if points_tensor is not None:
            pts = points_tensor
            # accept [N,2] or [1,N,2]
            if pts.ndim == 2 and pts.shape[-1] == 2:
                pts = pts.unsqueeze(0)  # -> [1,N,2]
            elif pts.ndim == 3 and pts.shape[-1] == 2:
                pass
            else:
                raise ValueError(f"points_tensor must be [N,2] or [1,N,2], got shape {tuple(points_tensor.shape)}")
            pts = pts.to(dtype=torch.float32, device=bundle.device)
            queries = torch.cat(
                [torch.ones_like(pts[:, :, :1]) * float(grid_query_frame), pts],
                dim=2,
            )  # [1,N,3]
            return queries

        # 2) grid mode
        if use_grid:
            grid_pts = get_points_on_a_grid(grid_size, (h, w))  # (1,M,2)
            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * float(grid_query_frame), grid_pts],
                dim=2,
            )  # (1,M,3)
            return queries.to(device=bundle.device, dtype=torch.float32)

        # 3) manual points from text (fallback to small grid if empty)
        pts_np = self._parse_points(points_text)
        if pts_np is None or pts_np.size == 0:
            print("[LiteTracker] No valid points_text; falling back to grid (size=10).")
            grid_pts = get_points_on_a_grid(max(10, grid_size or 10), (h, w))
            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * float(grid_query_frame), grid_pts],
                dim=2,
            )
            return queries.to(device=bundle.device, dtype=torch.float32)

        pts = torch.from_numpy(pts_np).unsqueeze(0)  # (1,N,2)
        queries = torch.cat(
            [torch.ones_like(pts[:, :, :1], dtype=pts.dtype) * float(grid_query_frame), pts],
            dim=2,
        )  # (1,N,3)
        return queries.to(device=bundle.device, dtype=torch.float32)

    def _prepare_segm_mask(self, video_uint8_bhwc: torch.Tensor, segm_mask_img: Optional[torch.Tensor]):
        """
        Convert an IMAGE mask (float [T,H,W,C] in 0..1) to (1, T, H, W) uint8 binary.
        Accepts single-frame mask [1,H,W,C] and broadcasts across T.
        """
        if segm_mask_img is None:
            return None
        if not isinstance(segm_mask_img, torch.Tensor) or segm_mask_img.ndim != 4:
            print(f"[LiteTracker] segm_mask must be IMAGE [T,H,W,C]; got {type(segm_mask_img)} shape {getattr(segm_mask_img, 'shape', None)}")
            return None

        T_v, H_v, W_v, _ = video_uint8_bhwc.shape
        T_m, H_m, W_m, _ = segm_mask_img.shape

        # take first channel if multi-channel
        mask = segm_mask_img[..., 0]  # [T,H,W]

        # broadcast if single-frame
        if T_m == 1 and T_v > 1:
            mask = mask.repeat(T_v, 1, 1)

        # resize if needed
        if (mask.shape[-2] != H_v) or (mask.shape[-1] != W_v):
            m = mask.unsqueeze(1)  # [T,1,H,W]
            m = torch.nn.functional.interpolate(m, size=(H_v, W_v), mode="nearest")
            mask = m.squeeze(1)

        # binarize and add batch dim
        mask = (mask >= 0.5).to(torch.uint8)  # [T,H,W]
        return mask.unsqueeze(0)              # [1,T,H,W]

    def track(
        self,
        tracker: LiteTrackerBundle,
        image: torch.Tensor,
        # keep exact order with INPUT_TYPES optional keys
        reset=False,
        use_grid=True,
        grid_size=10,
        grid_query_frame=0,
        points_text="",
        points_tensor=None,
        segm_mask=None,
        batch_is_sequence=True,
        preview_mode="visualizer",
        vis_also_save_file=False,
        vis_save_dir=os.path.join(os.path.dirname(__file__), "results"),
        vis_filename="",
        vis_fps=24.0,
        vis_mode="rainbow",
        vis_linewidth=2,
        vis_show_first_frame=0,
        vis_tracks_leave_trace=0,
        vis_pad_value=0,
        vis_grayscale=False,
        vis_opacity=1.0,
        vis_query_frame=0,
        vis_compensate_for_camera_motion=False,
    ):
        bundle = tracker

        # reset/init
        if reset or (not bundle.is_initialized):
            bundle.model.init_video_online_processing()
            bundle.is_initialized = False
            bundle.queries = None

        # convert to uint8 frames [T,H,W,C]
        img_uint8 = _to_uint8_bhwc(image)

        # build queries once
        if not bundle.is_initialized:
            queries = self._build_queries(
                bundle=bundle,
                img_uint8_bhwc=img_uint8,
                use_grid=use_grid,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                points_text=points_text,
                points_tensor=points_tensor,
            )
            bundle.queries = queries
            bundle.is_initialized = True

        # run online over frames
        device_type = "cuda" if bundle.device == "cuda" else ("mps" if bundle.device == "mps" else "cpu")
        use_autocast = (bundle.device == "cuda")
        coords_all, vis_all, preview_imgs = [], [], []

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=bundle.dtype if use_autocast else torch.float32, enabled=use_autocast):
                if batch_is_sequence:
                    for t in range(img_uint8.shape[0]):
                        frame = img_uint8[t:t+1]
                        frame_chw = frame.permute(0, 3, 1, 2).to(bundle.device, dtype=torch.float32)
                        coords, vis, conf = bundle.model(frame_chw, queries=bundle.queries)
                        c = coords[0, 0].detach().cpu()
                        v = vis[0, 0].detach().cpu() if vis.ndim == 3 else vis[0].detach().cpu()
                        coords_all.append(c); vis_all.append(v)
                        if preview_mode == "simple":
                            preview_imgs.append(_draw_points(frame.cpu(), [c.numpy()], [v.numpy()]))
                else:
                    frame = img_uint8[:1]
                    frame_chw = frame.permute(0, 3, 1, 2).to(bundle.device, dtype=torch.float32)
                    coords, vis, conf = bundle.model(frame_chw, queries=bundle.queries)
                    c = coords[0, 0].detach().cpu()
                    v = vis[0, 0].detach().cpu() if vis.ndim == 3 else vis[0].detach().cpu()
                    coords_all.append(c); vis_all.append(v)
                    if preview_mode == "simple":
                        preview_imgs.append(_draw_points(frame.cpu(), [c.numpy()], [v.numpy()]))

        coords_tensor = torch.stack(coords_all, dim=0) if len(coords_all) > 1 else coords_all[0].unsqueeze(0)
        vis_tensor = torch.stack(vis_all, dim=0) if len(vis_all) > 1 else vis_all[0].unsqueeze(0)

        preview = image
        video_path = ""

        # SIMPLE overlay
        if preview_mode == "simple" and len(preview_imgs) > 0:
            preview = torch.cat(preview_imgs, dim=0) if len(preview_imgs) > 1 else preview_imgs[0]

        # VISUALIZER overlay (no file saved unless vis_also_save_file=True)
        if preview_mode == "visualizer":
            if not _VIZ_IMPORT_OK:
                print(f"[LiteTracker] Visualizer import failed ({_VIZ_IMPORT_ERR}). Falling back to original frames.")
            else:
                video_uint8 = img_uint8.permute(0, 3, 1, 2).unsqueeze(0)  # [1,T,C,H,W]
                tracks_bt = coords_tensor.unsqueeze(0)                    # [1,T,N,2]
                vis_bt = vis_tensor.unsqueeze(0).unsqueeze(-1).to(torch.bool)  # [1,T,N,1]
                segm_bt = self._prepare_segm_mask(img_uint8, segm_mask)        # [1,T,H,W] or None

                do_cmc = bool(vis_compensate_for_camera_motion) and (segm_bt is not None)
                if vis_compensate_for_camera_motion and segm_bt is None:
                    print("[LiteTracker] camera-motion compensation requested but no segm_mask provided; disabling it.")

                fps_int = int(max(1, min(240, round(float(vis_fps)))))
                if not vis_filename:
                    import time as _time
                    vis_filename = f"litetracker_{int(_time.time())}"

                viz = Visualizer(
                    save_dir=vis_save_dir,
                    grayscale=vis_grayscale,
                    pad_value=vis_pad_value,
                    fps=fps_int,
                    mode=vis_mode,
                    linewidth=vis_linewidth,
                    show_first_frame=vis_show_first_frame,
                    tracks_leave_trace=vis_tracks_leave_trace,
                )
                res = viz.visualize(
                    video=video_uint8.to(torch.uint8),
                    tracks=tracks_bt,
                    visibility=vis_bt,
                    segm_mask=segm_bt,
                    filename=vis_filename,
                    query_frame=vis_query_frame,
                    save_video=bool(vis_also_save_file),
                    compensate_for_camera_motion=do_cmc,
                    opacity=float(vis_opacity),
                )
                preview = res[0].permute(0, 2, 3, 1).to(torch.float32) / 255.0
                preview = preview.contiguous().cpu()
                video_path = os.path.join(vis_save_dir, f"{vis_filename}.mp4") if vis_also_save_file else ""

        bundle.last_coords = coords_tensor
        bundle.last_vis = vis_tensor
        return (bundle, coords_tensor, vis_tensor, preview, video_path,)

NODE_CLASS_MAPPINGS = {
    "LiteTracker_LoadModel": LiteTracker_LoadModel,
    "LiteTracker_Track": LiteTracker_Track,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LiteTracker_LoadModel": "LiteTracker: Load Model",
    "LiteTracker_Track": "LiteTracker: Track",
}