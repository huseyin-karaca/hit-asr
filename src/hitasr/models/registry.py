__all__ = ['EXPERT_CLASSES', 'GROUP_A', 'GROUP_B', 'EXPERT_LABELS', 'register_expert', 'load_expert_classes']

EXPERT_CLASSES = {}


def register_expert(cls):
    EXPERT_CLASSES[cls.name] = cls
    return cls


def load_expert_classes():
    """Import every model module that the current kernel can import; return `{name: class}`."""
    import importlib
    for mod in ("hitasr.models.whisper", "hitasr.models.hf_ctc", "hitasr.models.parakeet",
                "hitasr.models.llm", "hitasr.models.nemo"):
        try:
            m = importlib.import_module(mod)
        except Exception:                                      # noqa: BLE001 — group not installed
            continue
        for obj in vars(m).values():
            if isinstance(obj, type) and getattr(obj, "name", None) and getattr(obj, "model_id", None) \
                    and hasattr(obj, "_encode_batch") and obj.__name__ != "ASRFrameExtractor":
                EXPERT_CLASSES.setdefault(obj.name, obj)
    return EXPERT_CLASSES


GROUP_A = ("whisper_large_v3", "whisper_large_v3_turbo", "distil_large_v3_5",
           "parakeet_tdt_0_6b_v3", "parakeet_ctc_1_1b", "granite_speech_4_1_2b",
           "qwen3_asr_1_7b", "voxtral_mini_3b", "cohere_transcribe", "kyutai_stt_2_6b",
           "hubert_large", "wav2vec2_large_robust", "whisper_base")
GROUP_B = ("parakeet_tdt_0_6b_v2", "canary_qwen_2_5b", "canary_1b_v2")

# Display names for tables and the manuscript.
EXPERT_LABELS = {
    "whisper_large_v3": "Whisper-large-v3", "whisper_large_v3_turbo": "Whisper-large-v3-turbo",
    "distil_large_v3_5": "Distil-Whisper-large-v3.5", "whisper_base": "Whisper-base",
    "parakeet_tdt_0_6b_v3": "Parakeet-TDT-0.6b-v3", "parakeet_tdt_0_6b_v2": "Parakeet-TDT-0.6b-v2",
    "parakeet_ctc_1_1b": "Parakeet-CTC-1.1b", "granite_speech_4_1_2b": "Granite-Speech-4.1-2b",
    "qwen3_asr_1_7b": "Qwen3-ASR-1.7B", "voxtral_mini_3b": "Voxtral-Mini-3B",
    "cohere_transcribe": "Cohere-Transcribe", "kyutai_stt_2_6b": "Kyutai-STT-2.6b",
    "hubert_large": "HuBERT-Large", "wav2vec2_large_robust": "Wav2Vec2-Large-Robust",
    "canary_qwen_2_5b": "Canary-Qwen-2.5b", "canary_1b_v2": "Canary-1b-v2",
}