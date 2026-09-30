__all__ = ['HookedLLMExtractor', 'GraniteSpeechExtractor', 'Qwen3ASRExtractor', 'VoxtralExtractor', 'KyutaiSTTExtractor', 'CohereTranscribeExtractor']

import numpy as np
import torch
import logging

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

from hitasr.extractor import ASRFrameExtractor, Encoded, Replay


class _EncoderDone(Exception):
    """Raised by `encode_only`'s pre-hook on the language model: the tower has run, the decoder is not wanted."""


def _stop_here(module, args):
    raise _EncoderDone


def _unwrap(out):
    """The `(B, T, D)` tensor inside whatever a module returned."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("last_hidden_state", "hidden_states"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, (tuple, list)):
        return _unwrap(out[0])
    if isinstance(out, dict):
        return _unwrap(next(iter(out.values())))
    raise TypeError(f"cannot find a tensor in {type(out).__name__}")


class HookedLLMExtractor(ASRFrameExtractor):
    """An audio-encoder + LLM transcriber: frames off the tower, transcript off `generate`.

    Subclasses set `model_cls` (or `auto_cls`), `tower_attr` (dotted path to
    the audio encoder; `None` = first submodule whose class name contains
    `Encoder` or `Tower`) and implement `_request(arrays)` -> processor inputs
    and `_decode_ids(out, inputs)` -> transcripts.
    """

    tower_attr = None
    decoder_attr = None             # dotted path to the language model; None = `get_decoder()` (for `encode_only`)
    request_state = ()              # attributes `_request` sets that `_decode_ids` reads (kept per `encode_only` batch)
    max_new_tokens = 256
    frame_rate_hz = None            # measured by smoke_test

    def _load(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self._model_cls().from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()
        self.tower = self._resolve_tower()
        print(f"  audio tower: {self.tower_attr} ({type(self.tower).__name__})")
        self._captured = []
        self.tower.register_forward_hook(
            lambda m, args, kwargs, out: self._captured.append((args, kwargs, out)), with_kwargs=True)

    def _model_cls(self):
        raise NotImplementedError

    def _resolve_tower(self):
        if self.tower_attr:
            m = self.model
            for part in self.tower_attr.split("."):
                m = getattr(m, part)
            return m
        for name, m in self.model.named_modules():
            cls = type(m).__name__
            if name and ("Encoder" in cls or "Tower" in cls) and "Layer" not in cls:
                self.tower_attr = name
                return m
        raise AttributeError(f"no audio encoder found on {type(self.model).__name__}; set tower_attr")

    @property
    def hidden_size(self):
        cfg = getattr(self.tower, "config", None)
        for attr in ("hidden_size", "d_model", "encoder_hidden_size"):
            if cfg is not None and hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        return int(getattr(self, "_captured_dim", 0))

    @property
    def num_encoder_layers(self):
        cfg = getattr(self.tower, "config", None)
        for attr in ("num_hidden_layers", "encoder_layers", "num_layers"):
            if cfg is not None and hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        return 0

    def _request(self, arrays):
        raise NotImplementedError

    def _decode_ids(self, out, inputs):
        raise NotImplementedError

    def _max_new_tokens(self, inputs):
        return self.max_new_tokens

    def _lengths(self, arrays, hidden, kwargs):
        """Valid frames per row: the tower's own mask when it matches `(B, T)`, else input lengths scaled."""
        out = self._captured[-1][2] if self._captured else None
        mask = getattr(out, "attention_mask", None) if not isinstance(out, torch.Tensor) else None
        if isinstance(mask, torch.Tensor) and mask.shape[:2] == hidden.shape[:2]:
            return mask.sum(-1).long().clamp(min=1)
        n = torch.tensor([len(a) for a in arrays], dtype=torch.float32)
        T = hidden.shape[1]
        return torch.ceil(n / n.max() * T).long().clamp(1, T)

    def _tower_hidden(self, arrays, inputs):
        """`(B, T, D)` frames and per-row lengths from what the hook caught during `generate`.

        The default covers a tower called once that returns a padded batch;
        towers that pack or stream override this.
        """
        if not self._captured:
            raise RuntimeError(f"{self.name}: the hook on {self.tower_attr} never fired during generate")
        args, kwargs, out = self._captured[0]
        hidden = _unwrap(out)
        if hidden.dim() != 3:
            raise RuntimeError(f"{self.name}: tower returned {tuple(hidden.shape)}, not (B, T, D); "
                               "override _tower_hidden for a packed or channels-first tower")
        return hidden, self._lengths(arrays, hidden, kwargs)

    def _prepared(self, arrays):
        inputs = self._request(arrays).to(self.device)
        # Audio goes into the tower, so it takes the tower's dtype — not the
        # model's: Kyutai keeps its Mimi codec in fp32 under a bf16 LLM.
        want = next(self.tower.parameters(), torch.empty(0, dtype=self.dtype)).dtype
        for k, v in inputs.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(want)
        return inputs

    def _generate(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=self._max_new_tokens(inputs), do_sample=False)

    def _encode_batch(self, arrays, layer):
        # One `generate` does both halves; the tower hook catches the frames.
        inputs = self._prepared(arrays)
        self._captured = []
        out = self._generate(inputs)
        hidden, lengths = self._tower_hidden(arrays, inputs)
        self._captured = []                          # drop the references to the tower outputs
        self._captured_dim = hidden.shape[-1]
        hyps = self._decode_ids(out, inputs)
        return Encoded(hidden, lengths.to(hidden.device), state=hyps)

    def _decode_batch(self, encoded):
        return encoded.state

    # ----------------------------------------------------------- deployment --

    def decoder_module(self):
        """The language model (the text decoder): the module `encode_only` stops `generate` at."""
        if self.decoder_attr:
            m = self.model
            for part in self.decoder_attr.split("."):
                m = getattr(m, part)
            return m
        get = getattr(self.model, "get_decoder", None)
        dec = get() if callable(get) else None
        if dec is None or dec is self.model or any(t is self.tower for t in dec.modules()):
            raise AttributeError(f"{self.name}: get_decoder() gives no language model apart from the tower; "
                                 "set decoder_attr")
        return dec

    def encode_only(self, arrays, layer=-1):
        """The tower alone. The same `generate` as extraction, stopped at the language model's first call: what ran
        is the processor, the tower and whatever the model's forward does between them, so the frames are the ones
        extraction stored. The tower's outputs stay in the batch's state for `decode_encoded` to replay."""
        self.load()
        arrays = self._pad_short([np.asarray(a, dtype=np.float32) for a in arrays])
        with self._grad_ctx():
            inputs = self._prepared(arrays)
            self._captured = []
            handle = self.decoder_module().register_forward_pre_hook(_stop_here)
            try:
                self._generate(inputs)
            except _EncoderDone:
                pass
            else:
                raise RuntimeError(f"{self.name}: generate finished without calling the language model")
            finally:
                handle.remove()
            hidden, lengths = self._tower_hidden(arrays, inputs)
            replay = [out for _, _, out in self._captured]
            self._captured = []
        self._captured_dim = hidden.shape[-1]
        request = {k: getattr(self, k) for k in self.request_state if hasattr(self, k)}
        return Encoded(hidden, lengths.to(hidden.device),
                       state={"inputs": inputs, "replay": replay, "request": request})

    def decode_encoded(self, encoded):
        """The transcripts of an `encode_only` batch: extraction's `generate`, with the tower's recorded outputs
        replayed in place of running the tower again."""
        st = encoded.state
        if not (isinstance(st, dict) and "replay" in st):
            return self.decode(encoded)                  # an `encode` batch carries its transcripts already
        for k, v in st["request"].items():
            setattr(self, k, v)
        with self._grad_ctx():
            self.tower.forward = Replay(st["replay"])
            try:
                out = self._generate(st["inputs"])
            finally:
                del self.tower.forward                   # the class's forward again
                self._captured = []
        return self._decode_ids(out, st["inputs"])


class GraniteSpeechExtractor(HookedLLMExtractor):
    """ibm-granite/granite-speech-4.1-2b: Conformer tower, Granite LLM. Needs `peft`."""
    name, model_id = "granite_speech_4_1_2b", "ibm-granite/granite-speech-4.1-2b"
    tower_attr = "model.encoder"

    def _model_cls(self):
        from transformers import GraniteSpeechForConditionalGeneration
        return GraniteSpeechForConditionalGeneration

    def _request(self, arrays):
        chat = [{"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}]
        text = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text] * len(arrays), audio=[np.asarray(a, dtype=np.float32) for a in arrays],
                              return_tensors="pt")

    def _decode_ids(self, out, inputs):
        n = inputs["input_ids"].shape[1]
        return [t.strip() for t in self.processor.batch_decode(out[:, n:], skip_special_tokens=True)]


class Qwen3ASRExtractor(HookedLLMExtractor):
    """Qwen/Qwen3-ASR-1.7B-hf: Qwen3-audio tower, Qwen3 LLM.

    The tower packs the valid post-CNN frames of every row into one
    `(sum T_i, D)` tensor (windowed attention over `cu_seqlens`); the per-row
    counts are the same function of the mel mask the processor uses to count
    audio tokens, so the split is exact.
    """
    name, model_id = "qwen3_asr_1_7b", "Qwen/Qwen3-ASR-1.7B-hf"
    tower_attr = "model.audio_tower"

    def _model_cls(self):
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM

    def _request(self, arrays):
        return self.processor.apply_transcription_request(
            audio=[np.asarray(a, dtype=np.float32) for a in arrays], language="English")

    def _tower_hidden(self, arrays, inputs):
        from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths
        args, kwargs, out = self._captured[0]
        packed = _unwrap(out)
        mask = kwargs.get("input_features_mask")
        if mask is None:
            mask = inputs["input_features_mask"]
        lengths = _get_feat_extract_output_lengths(mask.sum(-1).long(), self.tower.n_window).cpu()
        if int(lengths.sum()) != packed.shape[0]:
            raise RuntimeError(f"Qwen3-ASR packed {packed.shape[0]} frames but the mask implies "
                               f"{int(lengths.sum())}; the length formula no longer matches the encoder")
        hidden = torch.nn.utils.rnn.pad_sequence(list(torch.split(packed, lengths.tolist())), batch_first=True)
        return hidden, lengths.clamp(min=1)

    def _decode_ids(self, out, inputs):
        n = inputs["input_ids"].shape[1]
        raw = self.processor.batch_decode(out[:, n:], skip_special_tokens=True)
        return [t.strip() for t in self.processor.extract_transcription(raw)]


class VoxtralExtractor(HookedLLMExtractor):
    """mistralai/Voxtral-Mini-3B-2507: Whisper-style tower, Ministral LLM. Needs `mistral-common[audio]`.

    mistral-common cuts anything longer than 30 s into 30 s chunks and the
    tower sees one row per **chunk**, each padded to 1500 frames — so the
    hook's first axis is chunks, not utterances. Every chunk puts the same
    number of `[AUDIO]` tokens into its row's prompt, so the chunk count per
    utterance is read off `input_ids` and the chunks are re-joined along time.
    """
    name, model_id = "voxtral_mini_3b", "mistralai/Voxtral-Mini-3B-2507"
    tower_attr = "model.audio_tower"
    frame_rate_hz = 50.0

    def _model_cls(self):
        from transformers import VoxtralForConditionalGeneration
        return VoxtralForConditionalGeneration

    def _request(self, arrays):
        # Arrays are re-encoded through soundfile for mistral-common; WAV keeps them lossless.
        return self.processor.apply_transcription_request(
            language="en", audio=[np.asarray(a, dtype=np.float32) for a in arrays],
            model_id=self.model_id, sampling_rate=self.sampling_rate, format="wav")

    def _tower_hidden(self, arrays, inputs):
        args, kwargs, out = self._captured[0]
        chunks = _unwrap(out)                                                 # (n_chunks, 1500, D)
        n_tok = (inputs["input_ids"] == self.model.config.audio_token_id).sum(1).tolist()
        per_chunk = sum(n_tok) // chunks.shape[0]
        if per_chunk == 0 or sum(n_tok) != per_chunk * chunks.shape[0] or any(t % per_chunk for t in n_tok):
            raise RuntimeError(f"Voxtral: {chunks.shape[0]} tower rows but audio tokens per row {n_tok}")
        counts = [t // per_chunk for t in n_tok]
        joined = [c.reshape(-1, chunks.shape[-1]) for c in torch.split(chunks, counts)]   # (n_i * 1500, D)
        hidden = torch.nn.utils.rnn.pad_sequence(joined, batch_first=True)
        hop = self.sampling_rate / self.frame_rate_hz                                     # 320 samples a frame
        lengths = torch.tensor([min(max(1, -(-len(a) // int(hop))), j.shape[0]) for a, j in zip(arrays, joined)])
        return hidden, lengths

    def _decode_ids(self, out, inputs):
        n = inputs["input_ids"].shape[1]
        return [t.strip() for t in self.processor.batch_decode(out[:, n:], skip_special_tokens=True)]


class KyutaiSTTExtractor(HookedLLMExtractor):
    """kyutai/stt-2.6b-en-trfs: Mimi codec front end, streaming transformer. 24 kHz input.

    The frames are Mimi's pre-quantiser latent — the output of the codec's
    `downsample` layer, 512-d at 12.5 Hz — the last continuous acoustic
    representation before the codes the LLM reads. `generate` encodes the
    audio window by window with a padding cache, so the hook fires once per
    window and the windows are concatenated along time.
    """
    name, model_id = "kyutai_stt_2_6b", "kyutai/stt-2.6b-en-trfs"
    tower_attr = "codec_model.downsample"
    sampling_rate = 24000           # resampled from base's 16 kHz in `_request`
    source_rate = 16000

    def _model_cls(self):
        from transformers import KyutaiSpeechToTextForConditionalGeneration
        return KyutaiSpeechToTextForConditionalGeneration

    @property
    def hidden_size(self):
        return int(self.model.config.codec_config.hidden_size)

    @property
    def num_encoder_layers(self):
        return int(self.model.config.codec_config.num_hidden_layers)

    def _request(self, arrays):
        import librosa
        wavs = [librosa.resample(np.asarray(a, dtype=np.float32), orig_sr=self.source_rate,
                                 target_sr=self.sampling_rate) for a in arrays]
        return self.processor(audio=wavs, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)

    def _max_new_tokens(self, inputs):
        # One text token per 12.5 Hz audio frame: fewer would stop before the audio ends.
        return int(inputs["input_values"].shape[-1] // self.model.config.codec_config.frame_size)

    def _tower_hidden(self, arrays, inputs):
        windows = [_unwrap(out) for _, _, out in self._captured]            # each (B, D, t)
        hidden = torch.cat(windows, dim=-1).transpose(1, 2)                  # (B, T, D)
        # Mimi's conv arithmetic uses buffers that live on the device, so the lengths must too.
        n = torch.tensor([int(round(len(a) * self.sampling_rate / self.source_rate)) for a in arrays],
                         device=hidden.device)
        lengths = self.model.codec_model.get_encoded_length(n).long().clamp(1, hidden.shape[1]).cpu()
        return hidden, lengths

    def _decode_ids(self, out, inputs):
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]

    # `generate` streams the audio through Mimi window by window, interleaved with the language model, so it cannot
    # be stopped after the encoder: `encode_only` runs Mimi's encoder over the whole clip instead (the codec is
    # causal, so the latent is the streamed one), and `decode_encoded` runs the full streaming pipeline again.
    reuses_encoder = False

    def encode_only(self, arrays, layer=-1):
        self.load()
        arrays = self._pad_short([np.asarray(a, dtype=np.float32) for a in arrays])
        with self._grad_ctx():
            inputs = self._prepared(arrays)
            self._captured = []
            self.model.codec_model.encode(inputs["input_values"], padding_mask=inputs.get("padding_mask"))
            hidden, lengths = self._tower_hidden(arrays, inputs)
            self._captured = []
        return Encoded(hidden, lengths.to(hidden.device), state={"inputs": inputs})

    def decode_encoded(self, encoded):
        st = encoded.state
        if not (isinstance(st, dict) and "inputs" in st):
            return self.decode(encoded)
        with self._grad_ctx():
            out = self._generate(st["inputs"])
            self._captured = []
        return self._decode_ids(out, st["inputs"])


class CohereTranscribeExtractor(HookedLLMExtractor):
    """CohereLabs/cohere-transcribe-03-2026: Parakeet (FastConformer) encoder, Transformer decoder.

    The processor builds the decoder prompt itself, so `language` is
    required, and it cuts anything longer than 35 s at a quiet point near the
    boundary — so, as with Voxtral, the tower sees one row per **chunk**. The
    processor's `audio_chunk_index` maps rows back to utterances: the chunks'
    valid frames (off the encoder's own mask) are re-joined along time, and
    the transcripts joined in the same order.
    """
    name, model_id = "cohere_transcribe", "CohereLabs/cohere-transcribe-03-2026"
    tower_attr = "model.encoder"
    request_state = ("_chunk_index",)
    frame_rate_hz = 12.5            # 10 ms mel frames, subsampled x8
    language = "en"

    def _model_cls(self):
        from transformers import AutoModelForSpeechSeq2Seq
        return AutoModelForSpeechSeq2Seq

    def _request(self, arrays):
        inputs = self.processor(audio=[np.asarray(a, dtype=np.float32) for a in arrays], language=self.language,
                                sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        self._chunk_index = inputs.pop("audio_chunk_index")       # generate rejects it as a model kwarg
        return inputs

    def _tower_hidden(self, arrays, inputs):
        args, kwargs, out = self._captured[0]
        chunks = _unwrap(out)                                                 # (n_chunks, T, D)
        valid = out.attention_mask.sum(-1).tolist()
        rows = [[] for _ in arrays]
        for r, (i, _) in enumerate(self._chunk_index):
            rows[i].append(chunks[r, :valid[r]])
        joined = [torch.cat(r) for r in rows]
        hidden = torch.nn.utils.rnn.pad_sequence(joined, batch_first=True)
        lengths = torch.tensor([j.shape[0] for j in joined])
        return hidden, lengths

    def _decode_ids(self, out, inputs):
        n = inputs["decoder_input_ids"].shape[1]
        texts = self.processor.batch_decode(out[:, n:], skip_special_tokens=True)
        parts = [[] for _ in range(max(i for i, _ in self._chunk_index) + 1)]
        for (i, _), t in zip(self._chunk_index, texts):                      # rows come in chunk order
            parts[i].append(t.strip())
        return [" ".join(t for t in ps if t) for ps in parts]