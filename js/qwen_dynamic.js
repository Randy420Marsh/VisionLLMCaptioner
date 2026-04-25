/**
 * vision_llm_dynamic.js
 *
 * Dynamically manages image_N input slots on VisionLLMCaptioner.
 * Mode-aware: in "Text -> Detailed Image Prompt" mode all extra image slots are hidden.
 */

import { app } from "../../scripts/app.js";

console.log("[Gemma-4 Vision Captioner] JS extension loading…");

const TARGETS = ["VisionLLMCaptioner", "Qwen35RemoteCaptioner"]; // keep old name for compatibility

/** Extract N from "image_N", or -1. */
function imgNum(name) {
    const m = String(name ?? "").match(/^image_(\d+)$/);
    return m ? parseInt(m[1], 10) : -1;
}

/**
 * Main sync function – respects current Mode widget.
 * In Text mode → only image_1 exists.
 * In Image Caption mode → normal dynamic slots.
 */
function syncSlots(node) {
    const modeWidget = node.widgets ? node.widgets.find(w => w.name === "mode") : null;
    const isTextMode = modeWidget && String(modeWidget.value).includes("Text");

    let maxConnected = 0;
    const existing = new Set();

    for (const inp of node.inputs) {
        const n = imgNum(inp.name);
        if (n < 1) continue;
        existing.add(n);
        if (inp.link != null) maxConnected = Math.max(maxConnected, n);
    }

    if (isTextMode) {
        // Text mode: keep ONLY image_1
        for (let n = 2; n <= Math.max(...existing, 2); n++) {
            const idx = node.inputs.findIndex(i => i.name === `image_${n}`);
            if (idx !== -1) node.removeInput(idx);
        }
    } else {
        // Image Caption mode: dynamic slots
        const target = Math.max(1, maxConnected + 1);

        for (let n = 1; n <= target; n++) {
            if (!existing.has(n)) {
                node.addInput(`image_${n}`, "IMAGE");
                existing.add(n);
            }
        }

        const excess = [...existing].filter(n => n > target);
        excess.sort((a, b) => b - a);
        for (const n of excess) {
            const idx = node.inputs.findIndex(i => i.name === `image_${n}` && i.link == null);
            if (idx !== -1) node.removeInput(idx);
        }
    }

    app.graph?.setDirtyCanvas(true, true);
}

/** Manual: Add one extra slot (only in Image Caption mode) */
function addSlot(node) {
    const modeWidget = node.widgets ? node.widgets.find(w => w.name === "mode") : null;
    if (modeWidget && String(modeWidget.value).includes("Text")) return;

    let max = 0;
    for (const inp of node.inputs) {
        const n = imgNum(inp.name);
        if (n > max) max = n;
    }
    node.addInput(`image_${max + 1}`, "IMAGE");
    app.graph?.setDirtyCanvas(true, true);
}

/** Manual: Remove the highest unconnected slot */
function removeSlot(node) {
    const slots = node.inputs
        .filter(i => imgNum(i.name) > 1 && i.link == null)
        .sort((a, b) => imgNum(b.name) - imgNum(a.name));
    if (slots.length === 0) return;
    const idx = node.inputs.indexOf(slots[0]);
    if (idx !== -1) node.removeInput(idx);
    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "VisionLLM.MultiImageInputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGETS.includes(nodeData.name)) return;

        console.log(`[Gemma-4 Vision Captioner] Patching ${nodeData.name}`);

        const _origCC = nodeType.prototype.onConnectionChange;
        nodeType.prototype.onConnectionChange = function (side) {
            _origCC?.call(this, side);
            if (side === 1) syncSlots(this);
        };

        const _origCfg = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            _origCfg?.call(this, data);
            requestAnimationFrame(() => syncSlots(this));
        };
    },

    nodeCreated(node) {
        if (!TARGETS.includes(node.comfyClass)) return;

        console.log("[Gemma-4 Vision Captioner] Node created, adding controls");

        syncSlots(node);

        const modeWidget = node.widgets.find(w => w.name === "mode");
        if (modeWidget) {
            const originalCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (originalCallback) originalCallback.apply(this, arguments);
                syncSlots(node);
            };
        }

        node.addWidget("button", "＋  Add Image Slot", null, () => addSlot(node), { serialize: false });
        node.addWidget("button", "－  Remove Last Slot", null, () => removeSlot(node), { serialize: false });
    },
});

console.log("[Gemma-4 Vision Captioner] Extension registered ✓");
