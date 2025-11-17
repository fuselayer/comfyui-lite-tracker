Markdown

# ComfyUI LiteTracker

ComfyUI custom nodes for **LiteTracker**, a fast and efficient point tracking model for videos.

This implementation wraps the original [ImFusionGmbH/lite-tracker](https://github.com/ImFusionGmbH/lite-tracker) repository as ComfyUI nodes.

## Features

- **Load Model** node: Download and load LiteTracker weights
- **Track** node: Track points across video frames with online processing
- **Grid Editor** node: Graphical interface to define multiple rotatable tracking regions with custom point density

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:

```
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
Restart ComfyUI
The model weights will be automatically downloaded on first use.
```

Usage

Add LiteTracker: Load Model to load the tracking model
Add LiteTracker: Grid Editor to define tracking regions on your input image
Add LiteTracker: Track to process your video sequence
See the original LiteTracker repository for details on the underlying algorithm.

Credits
Original model and research: ImFusionGmbH/lite-tracker

ComfyUI integration: This repository