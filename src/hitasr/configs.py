"""The configurations the paper reports, fixed and named; the notebooks import them.

`MAIN[<corpus>]` is the main experiment on one corpus (`notebooks/main_<corpus>`): its expert trio, the protocol, the
reported 5x2 repetitions. `SPACES` is Table 2, the hyperparameter spaces every corpus searches. `ABLATION`, `TIMING`
and `SYNTHETIC` belong to the other three notebooks. A record written by a notebook carries its configuration.
"""

__all__ = ['SELECTION_ARMS', 'BASELINE_ARMS', 'CONTROL_ARMS', 'ARMS', 'TORCH_ARMS', 'SPACES', 'MainConfig', 'MAIN',
           'AblationConfig', 'ABLATION', 'TimingConfig', 'TIMING', 'SyntheticConfig', 'SYNTHETIC', 'RebuildConfig',
           'REBUILD', 'CORPUS_TITLES']

from dataclasses import dataclass, field

from labkit.search import Param

SELECTION_ARMS = ("hit_asr", "mlp_pool", "adastt_ce")      # HIT-ASR, the pooled router, the tree router (ADASTT)
BASELINE_ARMS = ("rover_conf", "cn_mbr")                     # the transcript-fusion baselines
CONTROL_ARMS = ("mlp_pool_hard_ce",)                         # the pooled router on hard CE alone (read by the ablation)
ARMS = SELECTION_ARMS + BASELINE_ARMS + CONTROL_ARMS
TORCH_ARMS = ("hit_asr", "mlp_pool", "mlp_pool_hard_ce")

CORPUS_TITLES = {"ami_sdm": "AMI-SDM", "earnings22": "Earnings-22", "peoples_speech": "People's Speech",
                 "afrispeech": "AfriSpeech-200"}


def _spaces(max_restarts=0):
    """Table 2. A parameter with `frozen=` is fixed at that value for every trial; the others are searched."""
    mlp = lambda loss: {p.name: p for p in [                                             # noqa: E731
        Param('d_hidden', 'cat', choices=(256, 512, 1024, 2048)),
        Param('n_layers', 'cat', choices=(1, 2, 3)),
        Param('dropout', 'cat', choices=(0.05, 0.15, 0.3)),
        Param('lr', 'cat', choices=(3e-05, 0.0001, 0.0003, 0.001)),
        Param('weight_decay', 'cat', choices=(0.001, 0.01, 0.1)),
        Param('batch_size', 'cat', choices=(32, 64, 128, 256)),
        Param('patience', 'cat', choices=(5, 10)),
        Param('loss_preset', 'cat', choices=('default', 'wer_only', 'hard_ce_only', 'soft_ce_only', 'wer+hard', 'all',
                                              'wer+soft_tau1'), frozen=loss),
        Param('label_smoothing', 'cat', choices=(0.0, 0.1)),
        Param('word_weighted', 'cat', choices=(True, False), frozen=False),
        Param('n_seeds', 'cat', choices=(1, 3), frozen=3),
    ]}
    spaces = {
        "hit_asr": {p.name: p for p in [
            Param('d_model', 'cat', choices=(128, 256)),
            Param('n_heads', 'cat', choices=(4, 8)),
            Param('stage1_layers', 'cat', choices=(1, 2, 3)),
            Param('stage2_layers', 'cat', choices=(1, 2)),
            Param('ffn_dim', 'cat', choices=(256, 512, 1024)),
            Param('dropout', 'cat', choices=(0.05, 0.15)),
            Param('fusion', 'cat', choices=('bridge', 'self_attn', 'concat', 'mean')),
            Param('share_stage1', 'cat', choices=(True, False)),
            Param('input_norm', 'cat', choices=('none', 'scale', 'standardize')),
            Param('pooled_skip', 'cat', choices=(False, True), frozen=False),
            Param('lr', 'cat', choices=(0.0001,)),
            Param('weight_decay', 'cat', choices=(0.001, 0.01, 0.1)),
            Param('batch_size', 'cat', choices=(8, 16, 32, 64)),
            Param('max_frames', 'cat', choices=(500, 1000, 2000)),
            Param('patience', 'cat', choices=(5, 10)),
            Param('loss_preset', 'cat', choices=('hard_ce_only', 'soft_ce_only', 'wer+hard', 'all')),
            Param('label_smoothing', 'cat', choices=(0.0, 0.1)),
            Param('word_weighted', 'cat', choices=(True, False), frozen=False),
            Param('n_seeds', 'cat', choices=(1, 3), frozen=3),
        ]},
        "mlp_pool": mlp('all'),
        "adastt_ce": {p.name: p for p in [
            Param('n_estimators', 'cat', choices=(100, 400, 1000)),
            Param('learning_rate', 'cat', choices=(0.01, 0.03, 0.1, 0.3)),
            Param('max_depth', 'cat', choices=(3, 5, 8)),
            Param('subsample', 'cat', choices=(0.6, 0.8, 1.0)),
            Param('colsample_bytree', 'cat', choices=(0.1, 0.3, 0.6, 1.0)),
            Param('min_child_weight', 'cat', choices=(1.0, 4.0, 16.0)),
            Param('reg_lambda', 'cat', choices=(0.3, 3.0, 30.0)),
            Param('objective', 'cat', choices=('expected_wer', 'cross_entropy'), frozen='cross_entropy'),
        ]},
        "rover_conf": {p.name: p for p in [
            Param('alpha', 'cat', choices=(0.0,)),
            Param('conf_source', 'cat', choices=('cn', 'probe', 'blend')),
            Param('weighting', 'cat', choices=('uniform', 'inverse_wer', 'log_odds')),
            Param('weight_by_conf', 'cat', choices=(False, True)),
            Param('conf_temp', 'cat', choices=(2.0,)),
            Param('null_penalty', 'cat', choices=(0.25, 0.5)),
        ]},
        "cn_mbr": {p.name: p for p in [
            Param('decode', 'cat', choices=('slot', 'hyp')),
            Param('candidates', 'cat', choices=('members+consensus',)),
            Param('prior', 'cat', choices=('uniform', 'inverse_wer', 'log_odds')),
            Param('temperature', 'cat', choices=(10.0,)),
            Param('null_prior', 'cat', choices=(0.05, 0.2)),
            Param('use_conf', 'cat', choices=(False, True)),
            Param('conf_source', 'cat', choices=('probe',)),
        ]},
        "mlp_pool_hard_ce": mlp('hard_ce_only'),
    }
    for a in TORCH_ARMS:                     # the trainer's restart guard: off, one fit per seed for every torch arm
        spaces[a]["max_restarts"] = Param("max_restarts", "cat", choices=(0, 1, 2), frozen=max_restarts)
    return spaces


SPACES = _spaces()


@dataclass(frozen=True)
class MainConfig:
    """The main experiment on one corpus. Written into its record (`main_<corpus>.json`) verbatim."""

    dataset: str
    members: tuple                     # the expert trio
    cv_seeds: tuple                    # the five 5x2 repetitions, in the order Tables 3 and 6 use them
    first_fold: int = 0                # the fold of the first repetition that enters the test statistic (see the paper)

    holdout_frac: float = 0.25         # share of the rows the search may see at all
    inner_val_frac: float = 0.30       # share of THAT scored rather than fitted
    epochs: int = 60                   # the torch arms' epoch ceiling; early stopping ends most fits sooner
    max_restarts: int = 0              # the trainer's restart guard (off)
    search_budget: int = 10            # random draws per arm, from `SPACES`
    arm_versions: dict = field(default_factory=lambda: {"hit_asr": "v2", "mlp_pool": "v2", "mlp_pool_hard_ce": "v2"})

    partition_seed: int = 20260901     # the hold-out cut
    study_seed: int = 7                # the search's sampler
    model_seed: int = 42               # model initialisation
    cv_random_seed: int = 20260821     # the random reference arm
    nthread: int = 4
    alpha: float = 0.05

    @property
    def title(self):
        return CORPUS_TITLES.get(self.dataset, self.dataset)


MAIN = {c.dataset: c for c in (
    MainConfig("ami_sdm", ("cohere_transcribe", "granite_speech_4_1_2b", "parakeet_tdt_0_6b_v3"),
               cv_seeds=(17, 11, 22, 21, 99), first_fold=0),
    MainConfig("earnings22", ("cohere_transcribe", "kyutai_stt_2_6b", "parakeet_tdt_0_6b_v3"),
               cv_seeds=(123456, 17, 99, 54321, 8675309), first_fold=1),
    MainConfig("peoples_speech", ("granite_speech_4_1_2b", "parakeet_ctc_1_1b", "parakeet_tdt_0_6b_v3"),
               cv_seeds=(123, 7, 256, 512, 1234), first_fold=1),
    MainConfig("afrispeech", ("cohere_transcribe", "qwen3_asr_1_7b", "voxtral_mini_3b"),
               cv_seeds=(2024, 123, 777, 3, 86400), first_fold=1),
)}


@dataclass(frozen=True)
class AblationConfig:
    """The ablation notebook: HIT-ASR's design switches on the hold-out, and the pool-size study."""

    corpora: tuple = ("ami_sdm", "earnings22", "peoples_speech")      # the trios the hold-out exploration covers
    collapse: float = 0.01             # closing at most this share of the gap = stayed on the best single expert
    k_corpora: tuple = ("peoples_speech",)
    ks: tuple = (3, 5, 10)
    k_seeds: tuple = (42, 43, 44)      # model seeds; each fit is the main configuration (its own 3-fit ensemble)
    candidates: tuple = ("cohere_transcribe", "distil_large_v3_5", "granite_speech_4_1_2b", "kyutai_stt_2_6b",
                         "parakeet_ctc_1_1b", "parakeet_tdt_0_6b_v3", "qwen3_asr_1_7b", "voxtral_mini_3b",
                         "whisper_large_v3", "whisper_large_v3_turbo")
    evict_frames: bool = True          # free each corpus's frames after its study (K=10 is ~70 GB of frames)


@dataclass(frozen=True)
class TimingConfig:
    """The timing notebook: every encoder, the router and one decoder, on raw audio, one clip at a time."""

    corpora: tuple = ("ami_sdm", "earnings22", "peoples_speech", "afrispeech")
    n_clips: int = 100                 # timed clips per corpus
    warmup: int = 3                    # untimed clips run first, per expert and per system
    clip_seed: int = 0                 # which clips of the router's scored half
    router_fold: int = 0               # the deployed router is this fold of a reported repetition
    fit_router: bool = True            # fit the fold's router when none is stored
    n_frame_checks: int = 20           # clips whose live frames are compared with the stored ones
    throughput_batch: int = 16         # the batch the labelling-pass estimate is measured at
    scaling_ks: tuple = (3, 5, 10, 20)  # the router alone against the number of experts


@dataclass(frozen=True)
class SyntheticConfig:
    """The synthetic regime-switch check."""

    n: int = 10000                     # clips (80 % train, 20 % test)
    T: int = 128                       # frames per clip
    n_regimes: int = 4
    ks: tuple = (3, 5, 10)             # experts
    seeds: tuple = (0, 1, 2)           # generator and model seeds; the tables are mean ± sd over them
    epochs: int = 50
    hit_asr: dict = field(default_factory=lambda: {
        "d_model": 256, "n_heads": 4, "stage1_layers": 2, "stage2_layers": 1, "ffn_dim": 512, "dropout": 0.15,
        "lr": 1e-4, "weight_decay": 1e-2, "batch_size": 32, "max_frames": 2000, "patience": 10,
        "label_smoothing": 0.1, "n_seeds": 1})
    mlp_pool: dict = field(default_factory=lambda: {
        "d_hidden": 1024, "n_layers": 2, "dropout": 0.15, "lr": 1e-4, "weight_decay": 1e-2, "batch_size": 64,
        "patience": 10, "n_seeds": 1})


@dataclass(frozen=True)
class RebuildConfig:
    """The extraction notebook (level 3): every trio expert of every corpus decodes it in full."""

    corpora: tuple = ("ami_sdm", "earnings22", "peoples_speech", "afrispeech")
    # the starting batch per expert on a 96 GB GPU; the longest clips are pushed through once and it is halved until
    # they fit
    batch: dict = field(default_factory=lambda: {
        "default": 128, "granite_speech_4_1_2b": 32, "qwen3_asr_1_7b": 32, "voxtral_mini_3b": 32,
        "kyutai_stt_2_6b": 32, "cohere_transcribe": 64})
    min_batch: int = 4
    # gated on the Hub: downloading them needs a (read) token of an account that accepted their terms
    gated: tuple = ("cohere_transcribe",)


ABLATION = AblationConfig()
TIMING = TimingConfig()
SYNTHETIC = SyntheticConfig()
REBUILD = RebuildConfig()
