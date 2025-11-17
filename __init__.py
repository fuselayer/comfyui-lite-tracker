# Register the LiteTracker nodes and the GUI Grid Editor node.
# Robust import so missing dicts won't break registration.
import importlib

from .nodes_lite_tracker import (
    NODE_CLASS_MAPPINGS as _LTT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _LTT_NAMES,
)

NODE_CLASS_MAPPINGS = dict(_LTT_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS = dict(_LTT_NAMES)

# Try to add the GUI grid editor node (RectEditor)
try:
    rect_mod = importlib.import_module(".nodes_rect_editor", __package__)
    rect_classes = getattr(rect_mod, "NODE_CLASS_MAPPINGS", None)
    rect_names = getattr(rect_mod, "NODE_DISPLAY_NAME_MAPPINGS", None)
    if rect_classes and rect_names:
        NODE_CLASS_MAPPINGS.update(rect_classes)
        NODE_DISPLAY_NAME_MAPPINGS.update(rect_names)
    else:
        RectEditor = getattr(rect_mod, "RectEditor", None)
        if RectEditor is not None:
            NODE_CLASS_MAPPINGS["RectEditor"] = RectEditor
            NODE_DISPLAY_NAME_MAPPINGS["RectEditor"] = "LiteTracker: Grid Editor"
except Exception as e:
    print("[LiteTracker] RectEditor not loaded detail:", repr(e))

# Front-end assets directory for the editor widget
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]