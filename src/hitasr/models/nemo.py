__all__ = ['NEMO_MIN_SAMPLES', 'ParakeetNeMoExtractor', 'CanaryQwenExtractor', 'Canary1bV2Extractor']

import os
import tempfile

import numpy as np
import torch

from hitasr.extractor import ASRFrameExtractor, Encoded

NEMO_MIN_SAMPLES = 1600          # 0.1 s: NeMo's per-feature normaliser rejects one-frame inputs


class ParakeetNeMoExtractor(ASRFrameExtractor):
    """nvidia/parakeet-tdt-0.6b-v2 through NeMo, batched."""

    name = "parakeet_tdt_0_6b_v2"
    model_id = "nvidia/parakeet-tdt-0.6b-v2"
    min_input_samples = NEMO_MIN_SAMPLES
    frame_rate_hz = 12.5
    grad_mode = "no_grad"
    group = "B"

    def __init__(self, **kw):
        kw.setdefault("dtype", torch.float32)          # in-graph STFT
        super().__init__(**kw)
        self._direct_decode = True

    def _load(self):
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id).to(self.device).eval()
        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except AttributeError:
            pass
        try:                                     # the CUDA-graph decoder re-capture fault; see fastt
            from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
            WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self.model, attribute_path="decoding.decoding")
        except Exception:                                      # noqa: BLE001
            pass

    @property
    def hidden_size(self):
        enc = self.model.encoder
        for attr in ("d_model", "_feat_out", "output_dim"):
            if hasattr(enc, attr):
                return int(getattr(enc, attr))
        return 1024

    @property
    def num_encoder_layers(self):
        return len(self.model.encoder.layers)

    def _encode_batch(self, arrays, layer):
        sig = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True).to(self.device)
        sig_len = torch.tensor([len(a) for a in arrays], device=self.device)
        feats, feat_len = self.model.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = self.model.encoder(audio_signal=feats, length=feat_len)
        enc_len = enc_len.long().clamp(min=1, max=enc.shape[-1])
        return Encoded(enc.transpose(1, 2), enc_len,
                       state={"enc": enc, "enc_len": enc_len, "arrays": list(arrays)})

    def _decode_batch(self, encoded):
        st = encoded.state
        if self._direct_decode:
            try:
                out = self.model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=st["enc"], encoded_lengths=st["enc_len"], return_hypotheses=False)
                if isinstance(out, tuple):
                    out = out[0]
                return [h.text if hasattr(h, "text") else str(h) for h in out]
            except (TypeError, AttributeError):
                self._direct_decode = False
        out = self.model.transcribe(audio=[np.asarray(a, dtype=np.float32) for a in st["arrays"]],
                                    batch_size=len(st["arrays"]), verbose=False)
        return [h.text if hasattr(h, "text") else str(h) for h in out]


class CanaryQwenExtractor(ASRFrameExtractor):
    """nvidia/canary-qwen-2.5b: FastConformer frames + SALM transcription."""

    name = "canary_qwen_2_5b"
    model_id = "nvidia/canary-qwen-2.5b"
    min_input_samples = NEMO_MIN_SAMPLES
    frame_rate_hz = 12.5
    grad_mode = "no_grad"
    group = "B"
    prompt = "Transcribe the following:"

    def _load(self):
        from nemo.collections.speechlm2.models import SALM
        self.model = SALM.from_pretrained(self.model_id).to(self.device).eval()
        self._perception = self.model.perception
        self._scratch = tempfile.mkdtemp(prefix="canary_")

    @property
    def hidden_size(self):
        enc = self._perception.encoder
        for attr in ("d_model", "_feat_out", "output_dim"):
            if hasattr(enc, attr):
                return int(getattr(enc, attr))
        return 1024

    @property
    def num_encoder_layers(self):
        return len(self._perception.encoder.layers)

    def _encode_batch(self, arrays, layer):
        sig = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in arrays], batch_first=True).to(self.device)
        sig_len = torch.tensor([len(a) for a in arrays], device=self.device)
        feats, feat_len = self._perception.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = self._perception.encoder(audio_signal=feats, length=feat_len)
        if enc.shape[1] == self.hidden_size and enc.shape[1] != enc.shape[2]:
            enc = enc.transpose(1, 2)
        return Encoded(enc, enc_len.long().clamp(min=1, max=enc.shape[1]), state=list(arrays))

    def _decode_batch(self, encoded):
        import soundfile as sf
        paths = []
        for i, a in enumerate(encoded.state):
            p = os.path.join(self._scratch, f"{i}.wav")
            sf.write(p, np.asarray(a, dtype=np.float32), self.sampling_rate)
            paths.append(p)
        try:
            prompts = [[{"role": "user", "content": f"{self.prompt} {self.model.audio_locator_tag}",
                         "audio": [p]}] for p in paths]
            ids = self.model.generate(prompts=prompts, max_new_tokens=256)
            return [self.model.tokenizer.ids_to_text(row.cpu()) for row in ids]
        finally:
            for p in paths:
                os.unlink(p)


class Canary1bV2Extractor(ParakeetNeMoExtractor):
    """nvidia/canary-1b-v2: FastConformer encoder, Transformer AED decoder, through `transcribe`."""

    name = "canary_1b_v2"
    model_id = "nvidia/canary-1b-v2"

    def _load(self):
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id).to(self.device).eval()
        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except AttributeError:
            pass
        self._direct_decode = False              # AED: always through `transcribe`

    def _decode_batch(self, encoded):
        st = encoded.state
        out = self.model.transcribe(audio=[np.asarray(a, dtype=np.float32) for a in st["arrays"]],
                                    batch_size=len(st["arrays"]), verbose=False,
                                    source_lang="en", target_lang="en", task="asr", pnc="yes")
        return [h.text if hasattr(h, "text") else str(h) for h in out]