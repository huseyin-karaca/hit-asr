__all__ = ['WhisperExtractor', 'WhisperLargeV3Extractor', 'WhisperLargeV3TurboExtractor', 'DistilLargeV35Extractor',
           'WhisperBaseExtractor']

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from hitasr.extractor import ASRFrameExtractor, Encoded


class WhisperExtractor(ASRFrameExtractor):
    """Any `WhisperForConditionalGeneration` checkpoint: encoder frames + greedy transcription."""

    name = "whisper_large_v3"
    model_id = "openai/whisper-large-v3"
    frame_rate_hz = 50.0
    samples_per_frame = 320          # 160-sample mel hop x 2 (conv stride)
    max_input_seconds = 30

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self.dtype).to(self.device).eval()
        self._gen_kwargs = dict(language="en", task="transcribe", num_beams=1,
                                max_new_tokens=200, max_length=None,
                                return_timestamps=False)

    @property
    def hidden_size(self):
        return self.model.config.d_model

    @property
    def num_encoder_layers(self):
        return self.model.config.encoder_layers

    def _encode_batch(self, arrays, layer):
        inputs = self.processor(arrays, sampling_rate=self.sampling_rate, return_tensors="pt")
        feats = inputs.input_features.to(self.device, self.dtype, non_blocking=True)
        enc = self.model.get_encoder()(feats, output_hidden_states=True)
        max_frames = enc.last_hidden_state.shape[1]
        lengths = torch.tensor(
            [max(1, min(max_frames, int(np.ceil(len(a) / self.samples_per_frame)))) for a in arrays],
            dtype=torch.long)
        return Encoded(enc.hidden_states[layer], lengths,
                       state={"input_features": feats, "encoder_outputs": enc})

    def _decode_batch(self, encoded):
        st = encoded.state
        try:
            ids = self.model.generate(input_features=st["input_features"],
                                      encoder_outputs=st["encoder_outputs"], **self._gen_kwargs)
        except (TypeError, ValueError):
            ids = self.model.generate(input_features=st["input_features"], **self._gen_kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)


class WhisperLargeV3Extractor(WhisperExtractor):
    name, model_id = "whisper_large_v3", "openai/whisper-large-v3"


class WhisperLargeV3TurboExtractor(WhisperExtractor):
    name, model_id = "whisper_large_v3_turbo", "openai/whisper-large-v3-turbo"


class DistilLargeV35Extractor(WhisperExtractor):
    name, model_id = "distil_large_v3_5", "distil-whisper/distil-large-v3.5"


class WhisperBaseExtractor(WhisperExtractor):
    """The manuscript's original Whisper-base — kept for the sensitivity check, not for the pool."""
    name, model_id = "whisper_base", "openai/whisper-base"
