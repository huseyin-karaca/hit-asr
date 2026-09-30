__version__ = "0.1.0"

import os as _os

# Read by huggingface_hub when it is first imported, so set here, before anything imports it: the frame shards are
# fetched with every core, and a slow Hub answer is waited for rather than failed at the 10 s default. On Colab the
# caches go to the runtime's local disk.
_os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
_os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
_os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
if _os.path.isdir("/content") and "HITASR_CACHE" not in _os.environ:
    _os.environ["HITASR_CACHE"] = "/content/hitasr_cache"
