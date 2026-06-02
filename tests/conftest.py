"""
Global test isolation.

This runs at import time — before any test module (or the app singleton in
api.routes.memory) is imported — so every storage path defaults to a throwaway
temp directory. No test can touch the real data/ dir, even tests that construct
a MemoryService without explicitly redirecting its stores.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="neuralvault-test-")
os.environ.setdefault("NEURALVAULT_DB_PATH", os.path.join(_TMP, "vectors.npy"))
os.environ.setdefault("NEURALVAULT_INDEX_PATH", os.path.join(_TMP, "memory_ivf"))
os.environ.setdefault("NEURALVAULT_METADATA_PATH", os.path.join(_TMP, "metadata.json"))
# Keep tests offline — never hit the live LLM.
os.environ.setdefault("ANTHROPIC_API_KEY", "")
