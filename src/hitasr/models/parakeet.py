__all__ = ['ParakeetTDTExtractor']

import numpy as np
from transformers import AutoModelForTDT, AutoProcessor

from hitasr.extractor import ASRFrameExtractor, Encoded, Replay


class ParakeetTDTExtractor(ASRFrameExtractor):
    """nvidia/parakeet-tdt-0.6b-v3 FastConformer frames + TDT transcription."""

    name = "parakeet_tdt_0_6b_v3"
    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    frame_rate_hz = 12.5

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForTDT.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()
        self._encoder = getattr(self.model, "encoder", None) or self.model.get_encoder()

    @property
    def _enc_cfg(self):
        return self.model.config.encoder_config

    @property
    def hidden_size(self):
        return self._enc_cfg.hidden_size

    @property
    def num_encoder_layers(self):
        return self._enc_cfg.num_hidden_layers

    def _encode_batch(self, arrays, layer):
        inputs = self.processor(audio=arrays, sampling_rate=self.sampling_rate, return_tensors="pt",
                                padding="longest", return_attention_mask=True)
        feats = inputs["input_features"].to(self.device, self.dtype, non_blocking=True)
        mask = inputs["attention_mask"].to(self.device, non_blocking=True)
        enc = self._encoder(input_features=feats, attention_mask=mask, output_hidden_states=True)
        enc_mask = getattr(enc, "attention_mask", None)
        if enc_mask is not None:
            lengths = enc_mask.sum(-1).long()
        else:
            f = self._enc_cfg.subsampling_factor
            lengths = ((mask.sum(-1) - 1) // f + 1).long()
        return Encoded(enc.hidden_states[layer], lengths.clamp(min=1),
                       state={"input_features": feats, "attention_mask": mask})

    def _decode_batch(self, encoded):
        st = encoded.state
        out = self.model.generate(input_features=st["input_features"], attention_mask=st["attention_mask"])
        ids = getattr(out, "sequences", out)
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    # ----------------------------------------------------------- deployment --
    # `generate` runs the encoder itself; `decode_encoded` hands it the output of `encode_only`'s pass instead.

    def encode_only(self, arrays, layer=-1):
        self.load()
        arrays = self._pad_short([np.asarray(a, dtype=np.float32) for a in arrays])
        layer = self.resolve_layer(layer)
        with self._grad_ctx():
            recorded = []
            handle = self._encoder.register_forward_hook(lambda m, a, out: recorded.append(out))
            try:
                enc = self._encode_batch(arrays, layer)
            finally:
                handle.remove()
        enc.state["encoder_out"] = recorded[0]
        return enc

    def decode_encoded(self, encoded):
        st = encoded.state
        if "encoder_out" not in st:
            return self.decode(encoded)
        with self._grad_ctx():
            self._encoder.forward = Replay([st["encoder_out"]])
            try:
                return self._decode_batch(encoded)
            finally:
                del self._encoder.forward