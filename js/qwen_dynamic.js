/**
 * vision_llm_dynamic.js — Fixed
 * FIX: Math.max spread on empty Set guarded.
 */
import { app } from "../../scripts/app.js";

console.log("[VisionLLMCaptioner] JS extension loading…");

const TARGETS = ["VisionLLMCaptioner", "Qwen35RemoteCaptioner"];

// Default presets (fallback if API unavailable)
const DEFAULT_PRESETS = {
    "Image Caption": {
        "Custom": "",
        "Default": (
            "You are a world-class visual analyst and master prompt engineer specializing in " +
            "photorealistic, cinematic image generation. " +
            "You have been given {image_count} image(s) labeled {image_labels}. " +
            "Always reference images strictly by their exact label in your internal thinking ONLY. " +
            "Output the final prompt only, no explanations."
        ),
    },
    "Text -> Detailed Image Prompt": {
        "Custom": "",
        "Default": (
            "You are a master prompt engineer specializing in detailed, " +
            "photorealistic image generation prompts. " +
            "Expand the input into a comprehensive, detailed prompt."
        ),
    },
    "Text to Text": {
        "Custom": "",
        "Default": (
            "You are a helpful text assistant. Process and transform the input text " +
            "according to the user's instructions."
        ),
    },
};

// Global presets cache
let PRESETS = JSON.parse(JSON.stringify(DEFAULT_PRESETS));

// API base URL
const API_BASE = "/vision_llmcaptioner";

// Load presets from API
async function loadPresetsFromAPI() {
    try {
        const response = await fetch(`${API_BASE}/presets`);
        if (response.ok) {
            const data = await response.json();
            PRESETS = data;
            console.log("[VisionLLMCaptioner] Loaded presets from API:", Object.keys(PRESETS));
            return true;
        }
    } catch (e) {
        console.log("[VisionLLMCaptioner] Could not load presets from API, using defaults:", e.message);
    }
    return false;
}

// Save preset to API
async function savePresetToAPI(mode, name, prompt) {
    try {
        const response = await fetch(`${API_BASE}/presets/save`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode, name, prompt }),
        });
        const data = await response.json();
        if (response.ok) {
            console.log("[VisionLLMCaptioner] Saved preset:", data);
            return true;
        } else {
            console.error("[VisionLLMCaptioner] Failed to save preset:", data.error);
            alert("Failed to save preset: " + data.error);
            return false;
        }
    } catch (e) {
        console.error("[VisionLLMCaptioner] Error saving preset:", e);
        alert("Error saving preset: " + e.message);
        return false;
    }
}

// Delete preset from API
async function deletePresetFromAPI(mode, name) {
    try {
        const response = await fetch(`${API_BASE}/presets/delete`, {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode, name }),
        });
        const data = await response.json();
        if (response.ok) {
            console.log("[VisionLLMCaptioner] Deleted preset:", data);
            return true;
        } else {
            console.error("[VisionLLMCaptioner] Failed to delete preset:", data.error);
            alert("Failed to delete preset: " + data.error);
            return false;
        }
    } catch (e) {
        console.error("[VisionLLMCaptioner] Error deleting preset:", e);
        alert("Error deleting preset: " + e.message);
        return false;
    }
}

function imgNum(name) {
    const m = String(name ?? "").match(/^image_(\d+)$/);
    return m ? parseInt(m[1], 10) : -1;
}

function updatePresets(node) {
    const modeWidget = node.widgets ? node.widgets.find(w => w.name === "mode") : null;
    const presetWidget = node.widgets ? node.widgets.find(w => w.name === "preset") : null;
    const systemPromptWidget = node.widgets ? node.widgets.find(w => w.name === "system_prompt") : null;

    if (!modeWidget || !presetWidget || !systemPromptWidget) return;

    const mode = String(modeWidget.value);
    const modePresets = PRESETS[mode] || {};

    // Get current preset value
    const currentPreset = String(presetWidget.value);

    // Update preset options for the current mode
    presetWidget.options.values = Object.keys(modePresets);

    // If current preset doesn't exist in new mode, reset to "Custom" or first available
    if (!modePresets.hasOwnProperty(currentPreset)) {
        presetWidget.value = "Custom";
    } else {
        presetWidget.value = currentPreset;
    }
}

function applyPreset(node) {
    const modeWidget = node.widgets ? node.widgets.find(w => w.name === "mode") : null;
    const presetWidget = node.widgets ? node.widgets.find(w => w.name === "preset") : null;
    const systemPromptWidget = node.widgets ? node.widgets.find(w => w.name === "system_prompt") : null;

    if (!modeWidget || !presetWidget || !systemPromptWidget) return;

    const mode = String(modeWidget.value);
    const preset = String(presetWidget.value);

    if (preset !== "Custom" && PRESETS[mode] && PRESETS[mode][preset] !== undefined) {
        systemPromptWidget.value = PRESETS[mode][preset];
        app.graph?.setDirtyCanvas(true, true);
    }
}

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
        // FIX: guard against empty set before spread
        const maxExisting = existing.size > 0 ? Math.max(...existing) : 1;
        for (let n = 2; n <= maxExisting; n++) {
            const idx = node.inputs.findIndex(i => i.name === `image_${n}`);
            if (idx !== -1) node.removeInput(idx);
        }
    } else {
        const target = Math.max(1, maxConnected + 1);

        for (let n = 1; n <= target; n++) {
            if (!existing.has(n)) {
                node.addInput(`image_${n}`, "IMAGE");
                existing.add(n);
            }
        }

        const excess = [...existing]
            .filter(n => n > target)
            .sort((a, b) => b - a);

        for (const n of excess) {
            const idx = node.inputs.findIndex(i => i.name === `image_${n}` && i.link == null);
            if (idx !== -1) node.removeInput(idx);
        }
    }

    app.graph?.setDirtyCanvas(true, true);
}

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

function removeSlot(node) {
    const slots = node.inputs
        .filter(i => imgNum(i.name) > 1 && i.link == null)
        .sort((a, b) => imgNum(b.name) - imgNum(a.name));
    if (slots.length === 0) return;
    const idx = node.inputs.indexOf(slots[0]);
    if (idx !== -1) node.removeInput(idx);
    app.graph?.setDirtyCanvas(true, true);
}

// Prompt for preset name
function promptPresetName(defaultName = "") {
    return new Promise((resolve) => {
        const name = prompt("Enter preset name:", defaultName);
        resolve(name);
    });
}

// Save current preset
async function savePreset(node) {
    const modeWidget = node.widgets ? node.widgets.find(w => w.name === "mode") : null;
    const presetWidget = node.widgets ? node.widgets.find(w => w.name === "preset") : null;
    const systemPromptWidget = node.widgets ? node.widgets.find(w => w.name === "system_prompt") : null;

    if (!modeWidget || !systemPromptWidget) return;

    const mode = String(modeWidget.value);
    const currentPrompt = systemPromptWidget.value || "";
    const currentPreset = presetWidget ? String(presetWidget.value) : "Custom";

    const name = await promptPresetName(currentPreset === "Custom" ? "" : currentPreset);
    if (!name) return; // Cancelled

    // Check for reserved names
    if (name === "Custom" || name === "Default") {
        alert("Cannot use 'Custom' or 'Default' as preset name.");
        return;
    }

    // Save via API
    const success = await savePresetToAPI(mode, name, currentPrompt);
    if (success) {
        // Reload presets from API
        await loadPresetsFromAPI();
        // Update dropdown
        updatePresets(node);
        // Select the new preset
        if (presetWidget && PRESETS[mode] && PRESETS[mode][name]) {
            presetWidget.value = name;
        }
        app.graph?.setDirtyCanvas(true, true);
    }
}

// Delete current preset
async function deletePreset(node) {
    const modeWidget = node.widgets ? node.widgets.find(w => w.name === "mode") : null;
    const presetWidget = node.widgets ? node.widgets.find(w => w.name === "preset") : null;

    if (!modeWidget || !presetWidget) return;

    const mode = String(modeWidget.value);
    const preset = String(presetWidget.value);

    // Don't delete default presets
    if (preset === "Custom" || preset === "Default") {
        alert("Cannot delete 'Custom' or 'Default' presets.");
        return;
    }

    if (!confirm(`Delete preset "${preset}"?`)) return;

    // Delete via API
    const success = await deletePresetFromAPI(mode, preset);
    if (success) {
        // Reload presets from API
        await loadPresetsFromAPI();
        // Update dropdown
        presetWidget.value = "Custom";
        updatePresets(node);
        app.graph?.setDirtyCanvas(true, true);
    }
}

// Refresh presets from API
async function refreshPresets(node) {
    await loadPresetsFromAPI();
    updatePresets(node);
    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "VisionLLM.MultiImageInputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGETS.includes(nodeData.name)) return;

        const _origCC = nodeType.prototype.onConnectionChange;
        nodeType.prototype.onConnectionChange = function (side) {
            _origCC?.call(this, side);
            if (side === 1) syncSlots(this);
        };

        const _origCfg = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            _origCfg?.call(this, data);
            requestAnimationFrame(() => {
                syncSlots(this);
                updatePresets(this);
            });
        };
    },

    nodeCreated(node) {
        if (!TARGETS.includes(node.comfyClass)) return;

        syncSlots(node);

        const modeWidget = node.widgets.find(w => w.name === "mode");
        if (modeWidget) {
            const originalCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (originalCallback) originalCallback.apply(this, arguments);
                syncSlots(node);
                updatePresets(node);
            };
        }

        const presetWidget = node.widgets.find(w => w.name === "preset");
        if (presetWidget) {
            const originalCallback = presetWidget.callback;
            presetWidget.callback = function () {
                if (originalCallback) originalCallback.apply(this, arguments);
                applyPreset(node);
            };
        }

        // Preset management buttons
        node.addWidget("button", "Save Preset", null, () => savePreset(node), { serialize: false });
        node.addWidget("button", "Delete Preset", null, () => deletePreset(node), { serialize: false });
        node.addWidget("button", "Refresh Presets", null, () => refreshPresets(node), { serialize: false });

        // Original image slot buttons
        node.addWidget("button", "＋  Add Image Slot",   null, () => addSlot(node),    { serialize: false });
        node.addWidget("button", "－  Remove Last Slot", null, () => removeSlot(node), { serialize: false });
    },
});

// Load presets on startup
loadPresetsFromAPI();

console.log("[VisionLLMCaptioner] Extension registered ✓");