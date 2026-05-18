from .VisionLLMCaptioner import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    register_preset_api,
    WEB_DIRECTORY
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Register preset API when ComfyUI initializes
try:
    import server
    register_preset_api(server)
except (ImportError, AttributeError):
    pass