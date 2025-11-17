const { app } = window.comfyAPI.app;

const MODAL_ID = "lt-grid-editor-modal";

function ensureModal() {
    let modal = document.getElementById(MODAL_ID);
    if (modal) return modal;

    modal = document.createElement("div");
    modal.id = MODAL_ID;
    Object.assign(modal.style, {
        position: "fixed",
        inset: "0",
        background: "rgba(0,0,0,0.6)",
        display: "none",
        zIndex: 10000,
    });

    const panel = document.createElement("div");
    Object.assign(panel.style, {
        position: "absolute",
        left: "50%",
        top: "50%",
        transform: "translate(-50%,-50%)",
        background: "#1e1e1e",
        border: "1px solid #555",
        padding: "10px",
        boxShadow: "0 0 12px #000",
        maxWidth: "95vw",
        maxHeight: "95vh",
    });

    const header = document.createElement("div");
    header.textContent = "LiteTracker: Grid Editor (Multi-Rectangle)";
    Object.assign(header.style, { marginBottom: "6px", fontWeight: "bold" });

    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 360;
    Object.assign(canvas.style, {
        background: "#111",
        border: "1px solid #666",
        cursor: "crosshair",
        maxWidth: "90vw",
        maxHeight: "75vh",
    });

    const controls = document.createElement("div");
    Object.assign(controls.style, {
        marginTop: "8px",
        display: "flex",
        gap: "8px",
        alignItems: "center",
        flexWrap: "wrap",
    });

    const btnUndo = document.createElement("button");
    btnUndo.textContent = "Undo";
    const btnClear = document.createElement("button");
    btnClear.textContent = "Clear All";
    const btnApply = document.createElement("button");
    btnApply.textContent = "Apply";
    const btnCancel = document.createElement("button");
    btnCancel.textContent = "Cancel";
    
    controls.append(btnUndo, btnClear, btnApply, btnCancel);

    panel.append(header, canvas, controls);
    modal.appendChild(panel);
    document.body.appendChild(modal);

    // --- State ---
    let rectangles = [];  // [{x, y, w, h, rotation}, ...]
    let currentRect = null;  // Rectangle being drawn
    let interactionMode = 'idle';  // 'idle' | 'drawing' | 'dragging' | 'rotating'
    let draggingIndex = -1;
    let rotatingIndex = -1;
    let dragOffset = { x: 0, y: 0 };  // Offset between mouse and rect origin when dragging starts
    let drawStartPos = { x: 0, y: 0 };  // Starting position when drawing new rect
    let lastNode = null;
    let bgImage = null;

    // --- Helper functions ---
    function getMousePos(e) {
        const r = canvas.getBoundingClientRect();
        // Scale mouse coordinates to match canvas internal dimensions
        const scaleX = canvas.width / r.width;
        const scaleY = canvas.height / r.height;
        return {
            x: (e.clientX - r.left) * scaleX,
            y: (e.clientY - r.top) * scaleY
        };
    }

    function getRotationHandlePos(rect) {
        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;
        const handleDist = Math.max(rect.h / 2, rect.w / 2) + 25;
        const hx = cx + handleDist * Math.sin(rect.rotation);
        const hy = cy - handleDist * Math.cos(rect.rotation);
        return { x: hx, y: hy };
    }

    function distance(x1, y1, x2, y2) {
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    }

    function pointInRotatedRect(px, py, rect) {
        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;
        const dx = px - cx;
        const dy = py - cy;
        const cos_a = Math.cos(-rect.rotation);
        const sin_a = Math.sin(-rect.rotation);
        const local_x = dx * cos_a - dy * sin_a;
        const local_y = dx * sin_a + dy * cos_a;
        return Math.abs(local_x) <= rect.w / 2 && Math.abs(local_y) <= rect.h / 2;
    }

    function drawRotatedRect(ctx, rect, isActive = false) {
        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rect.rotation);
        
        ctx.strokeStyle = isActive ? "#0ff" : "#0aa";
        ctx.lineWidth = 2;
        ctx.strokeRect(-rect.w / 2, -rect.h / 2, rect.w, rect.h);
        
        // Corner handles
        ctx.fillStyle = "#0ff";
        const r = 3;
        ctx.fillRect(-rect.w / 2 - r, -rect.h / 2 - r, r * 2, r * 2);
        ctx.fillRect(rect.w / 2 - r, -rect.h / 2 - r, r * 2, r * 2);
        ctx.fillRect(-rect.w / 2 - r, rect.h / 2 - r, r * 2, r * 2);
        ctx.fillRect(rect.w / 2 - r, rect.h / 2 - r, r * 2, r * 2);
        
        ctx.restore();
    }

    function drawRotationHandle(ctx, rect) {
        const handle = getRotationHandlePos(rect);
        ctx.fillStyle = "#ff0";
        ctx.beginPath();
        ctx.arc(handle.x, handle.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    function draw() {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background image
        if (bgImage && bgImage.complete) {
            ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
        } else {
            ctx.fillStyle = "#111";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        // Draw all finalized rectangles
        rectangles.forEach((rect, i) => {
            const isActive = (i === rotatingIndex || i === draggingIndex);
            drawRotatedRect(ctx, rect, isActive);
            drawRotationHandle(ctx, rect);
        });

        // Draw current rectangle being drawn
        if (currentRect) {
            drawRotatedRect(ctx, currentRect, true);
        }
    }

    // --- Mouse events ---
    canvas.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;  // Only left click
        const { x, y } = getMousePos(e);

        // Priority 1: Check rotation handles (smaller target, needs priority)
        for (let i = 0; i < rectangles.length; i++) {
            const handle = getRotationHandlePos(rectangles[i]);
            if (distance(x, y, handle.x, handle.y) < 15) {  // Increased hit radius
                interactionMode = 'rotating';
                rotatingIndex = i;
                draw();
                return;
            }
        }

        // Priority 2: Check if clicking inside existing rectangle (for dragging)
        for (let i = rectangles.length - 1; i >= 0; i--) {  // Reverse order for top-to-bottom
            if (pointInRotatedRect(x, y, rectangles[i])) {
                interactionMode = 'dragging';
                draggingIndex = i;
                dragOffset.x = x - rectangles[i].x;
                dragOffset.y = y - rectangles[i].y;
                draw();
                return;
            }
        }

        // Priority 3: Start drawing new rectangle (empty space clicked)
        interactionMode = 'drawing';
        drawStartPos.x = x;
        drawStartPos.y = y;
        currentRect = { x, y, w: 1, h: 1, rotation: 0 };
        draw();
    });

    canvas.addEventListener("mousemove", (e) => {
        const { x, y } = getMousePos(e);

        if (interactionMode === 'drawing' && currentRect) {
            // Update width/height based on mouse position
            const dx = x - drawStartPos.x;
            const dy = y - drawStartPos.y;
            
            // Handle negative dimensions
            if (dx >= 0) {
                currentRect.x = drawStartPos.x;
                currentRect.w = dx;
            } else {
                currentRect.x = x;
                currentRect.w = -dx;
            }
            
            if (dy >= 0) {
                currentRect.y = drawStartPos.y;
                currentRect.h = dy;
            } else {
                currentRect.y = y;
                currentRect.h = -dy;
            }
            
            draw();
        } else if (interactionMode === 'dragging' && draggingIndex >= 0) {
            const rect = rectangles[draggingIndex];
            rect.x = x - dragOffset.x;
            rect.y = y - dragOffset.y;
            draw();
        } else if (interactionMode === 'rotating' && rotatingIndex >= 0) {
            const rect = rectangles[rotatingIndex];
            const cx = rect.x + rect.w / 2;
            const cy = rect.y + rect.h / 2;
            rect.rotation = Math.atan2(x - cx, cy - y);
            draw();
        }
    });

    canvas.addEventListener("mouseup", (e) => {
        if (e.button !== 0) return;

        if (interactionMode === 'drawing' && currentRect) {
            // Only add if reasonably sized
            if (currentRect.w > 5 && currentRect.h > 5) {
                rectangles.push(currentRect);
            }
            currentRect = null;
        }

        interactionMode = 'idle';
        draggingIndex = -1;
        rotatingIndex = -1;
        draw();
    });

    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        const { x, y } = getMousePos(e);

        // Hit-test rectangles in reverse order (top to bottom)
        for (let i = rectangles.length - 1; i >= 0; i--) {
            if (pointInRotatedRect(x, y, rectangles[i])) {
                rectangles.splice(i, 1);
                draw();
                return;
            }
        }
    });

    // --- Buttons ---
    btnUndo.onclick = () => {
        if (rectangles.length > 0) {
            rectangles.pop();
            draw();
        }
    };

    btnClear.onclick = () => {
        rectangles = [];
        draw();
    };

    btnApply.onclick = () => {
        if (!lastNode) {
            modal.style.display = "none";
            return;
        }
        const wRect = lastNode.widgets?.find((w) => w.name === "rect_json");
        if (wRect) {
            wRect.value = JSON.stringify(rectangles);
            app.graph.setDirtyCanvas(true, true);
        }
        modal.style.display = "none";
    };

    btnCancel.onclick = () => {
        modal.style.display = "none";
    };

    // --- Modal show function ---
    modal._ltGridEditor = {
        show(node) {
            lastNode = node;
            bgImage = null;

            // Load background from node.properties
            if (node.properties && node.properties.imgData && node.properties.imgData.base64) {
                const base64Data = node.properties.imgData.base64;
                const im = new Image();
                im.onload = () => {
                    bgImage = im;
                    canvas.width = im.naturalWidth;
                    canvas.height = im.naturalHeight;
                    draw();
                };
                im.onerror = () => {
                    console.error("[LiteTracker] Failed to load background image");
                    bgImage = null;
                    draw();
                };
                console.log("[LiteTracker] Loading background, base64 length:", base64Data.length);
                im.src = `data:image/png;base64,${base64Data}`;
            } else {
                canvas.width = 640;
                canvas.height = 360;
                draw();
            }

            // Load existing rectangles
            const wRect = node.widgets?.find((w) => w.name === "rect_json");
            if (wRect && wRect.value) {
                try {
                    const parsed = JSON.parse(wRect.value);
                    if (Array.isArray(parsed)) {
                        rectangles = parsed;
                    } else if (parsed && typeof parsed === 'object') {
                        // Old single-rectangle format
                        rectangles = [parsed];
                    } else {
                        rectangles = [];
                    }
                } catch {
                    rectangles = [];
                }
            } else {
                rectangles = [];
            }

            modal.style.display = "block";
            draw();
        },
    };

    return modal;
}

// =====================================================================
//  Extension registration
// =====================================================================

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existent object");
        return;
    }
    if (property in object) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r;
        };
    } else {
        object[property] = callback;
    }
}

app.registerExtension({
    name: "comfyui-lite-tracker.grid-editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "RectEditor") {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                // Add open button
                this.addWidget("button", "Open Grid Editor", "open", () => {
                    try {
                        const modal = ensureModal();
                        modal._ltGridEditor.show(this);
                    } catch (e) {
                        console.error("[LiteTracker] Grid Editor: failed to open modal:", e);
                    }
                });

                // Store base64 in properties on execution
                chainCallback(this, "onExecuted", function (message) {
                    try {
                        const bg_base64 = message["bg_base64"];
                        if (bg_base64) {
                            if (!this.properties) {
                                this.properties = {};
                            }
                            this.properties.imgData = {
                                name: "bg_base64",
                                type: "image/png",
                                base64: Array.isArray(bg_base64) ? bg_base64[0] : bg_base64
                            };
                            console.log("[LiteTracker] Stored base64, length:", this.properties.imgData.base64.length);
                        }
                    } catch (e) {
                        console.warn("[LiteTracker] onExecuted error:", e);
                    }
                });
            });
        }
    },
});