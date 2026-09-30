__all__ = ['HFCtcExtractor', 'HubertLargeExtractor', 'Wav2Vec2LargeRobustExtractor', 'ParakeetCtcExtractor']

import torch
from transformers import AutoModelForCTC, AutoProcessor

from hitasr.extractor import ASRFrameExtractor, Encoded


class HFCtcExtractor(ASRFrameExtractor):
    """Any CTC speech checkpoint `transformers` exposes as `AutoModelForCTC`."""

    frame_rate_hz = 50.0

    def __init__(self, **kw):
        kw.setdefault("dtype", torch.float32)          # fp32 weights + bf16 autocast
        super().__init__(**kw)

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()
        fe = self.processor.feature_extractor
        self._pass_mask = bool(getattr(fe, "return_attention_mask", True))

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_encoder_layers(self):
        return self.model.config.num_hidden_layers

    def _encode_batch(self, arrays, layer):
        inputs = self.processor(arrays, return_tensors="pt", padding="longest",
                                sampling_rate=self.sampling_rate, return_attention_mask=True)
        key = "input_values" if "input_values" in inputs else "input_features"
        x = inputs[key].to(self.device, non_blocking=True)
        kw = {}
        if self._pass_mask and "attention_mask" in inputs:
            kw["attention_mask"] = inputs["attention_mask"].to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
            out = self.model(**{key: x}, output_hidden_states=True, **kw)
        T = out.hidden_states[layer].shape[1]
        helper = getattr(self.model, "_get_feat_extract_output_lengths", None)
        if helper is not None and "attention_mask" in inputs:
            lengths = helper(inputs["attention_mask"].sum(-1)).long().clamp(1, T)
        else:
            n = torch.tensor([len(a) for a in arrays], dtype=torch.float32)
            lengths = torch.ceil(n / n.max() * T).long().clamp(1, T)
        return Encoded(out.hidden_states[layer], lengths, state=out.logits)

    def _decode_batch(self, encoded):
        pred = encoded.state.float().argmax(-1).cpu()
        return [self.processor.decode(pred[i, :L]) for i, L in enumerate(encoded.lengths.tolist())]


class HubertLargeExtractor(HFCtcExtractor):
    name, model_id = "hubert_large", "facebook/hubert-large-ls960-ft"


class Wav2Vec2LargeRobustExtractor(HFCtcExtractor):
    name, model_id = "wav2vec2_large_robust", "facebook/wav2vec2-large-robust-ft-swbd-300h"


class ParakeetCtcExtractor(HFCtcExtractor):
    """nvidia/parakeet-ctc-1.1b: FastConformer frames at 12.5 Hz, CTC decode."""
    name, model_id = "parakeet_ctc_1_1b", "nvidia/parakeet-ctc-1.1b"
    frame_rate_hz = 12.5

    @property
    def hidden_size(self):
        cfg = self.model.config
        return getattr(getattr(cfg, "encoder_config", cfg), "hidden_size")

    @property
    def num_encoder_layers(self):
        cfg = self.model.config
        return getattr(getattr(cfg, "encoder_config", cfg), "num_hidden_layers")