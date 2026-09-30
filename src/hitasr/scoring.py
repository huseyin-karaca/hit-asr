__all__ = ['load_english_normalizer', 'WerScorer', 'corpus_wer_table']

import jiwer
import pandas as pd


def load_english_normalizer(model_id="openai/whisper-large-v3"):
    """The Whisper English text normaliser the model cards use, or a naive fallback."""
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError:
        print("WARNING: transformers not installed; falling back to a naive "
              "uppercase normaliser. `wer` will not match the model cards.")
        return lambda s: " ".join(s.upper().split())
    mapping = {}
    try:
        from transformers import WhisperTokenizer
        mapping = WhisperTokenizer.from_pretrained(model_id).english_spelling_normalizer
    except Exception as e:                                 # noqa: BLE001
        print(f"WARNING: could not load the spelling map ({type(e).__name__}); "
              "normalising without spelling harmonisation.")
    return EnglishTextNormalizer(mapping)


class WerScorer:
    """Word-level error counters, normalised and raw. `normalizer` is any `str -> str`."""

    def __init__(self, normalizer=None):
        self.normalizer = normalizer if normalizer is not None else load_english_normalizer()

    @staticmethod
    def _counters(ref, hyp):
        """(substitutions, deletions, insertions, reference word count)."""
        if not ref.strip():
            return 0, 0, len(hyp.split()), 0
        w = jiwer.process_words(ref, hyp)
        return w.substitutions, w.deletions, w.insertions, (w.hits + w.substitutions + w.deletions)

    def score(self, refs, hyps):
        """Score a batch. Returns a dict of columns ready to merge into a batch."""
        refs_n = [self.normalizer(r) for r in refs]
        hyps_n = [self.normalizer(h) for h in hyps]
        cols = {k: [] for k in ("sub", "dele", "ins", "nref", "wer",
                                "sub_raw", "dele_raw", "ins_raw", "nref_raw", "wer_raw")}
        for ref, hyp, ref_n, hyp_n in zip(refs, hyps, refs_n, hyps_n):
            for suffix, (r, h) in (("", (ref_n, hyp_n)), ("_raw", (ref, hyp))):
                s, d, i, n = self._counters(r, h)
                cols[f"sub{suffix}"].append(s)
                cols[f"dele{suffix}"].append(d)
                cols[f"ins{suffix}"].append(i)
                cols[f"nref{suffix}"].append(n)
                cols[f"wer{suffix}"].append((s + d + i) / n if n > 0 else 0.0)
        cols["text_norm"] = refs_n
        cols["transcription_norm"] = hyps_n
        return cols


def corpus_wer_table(frames):
    """Corpus WER per split from `{split: DataFrame}` of label rows. Check it before pushing."""
    rows = []
    for split, d in frames.items():
        row = {"split": split, "n_utts": len(d)}
        for suffix, label in (("", "norm"), ("_raw", "raw")):
            err = int(d[f"sub{suffix}"].sum() + d[f"dele{suffix}"].sum() + d[f"ins{suffix}"].sum())
            ref = int(d[f"nref{suffix}"].sum())
            row[f"wer_{label}"] = err / ref if ref else float("nan")
            row[f"errors_{label}"] = err
            row[f"ref_words_{label}"] = ref
        rows.append(row)
    return pd.DataFrame(rows)