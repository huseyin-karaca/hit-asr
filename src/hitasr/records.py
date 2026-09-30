"""The records the notebooks read: `results/<corpus>/<name>.json` and `results/<name>.json` on the Hub.

Set `HITASR_RECORDS_DIR` to a folder holding records you wrote yourself (a level-2 run of `main_<corpus>` writes
`main_<corpus>.json` next to the notebook): a record found there is read instead of the Hub's.
"""

__all__ = ['load_record']

import json
import os
from pathlib import Path

from hitasr.hub import HitHub


def load_record(name, corpus=None):
    """`results/<corpus>/<name>.json` (or `results/<name>.json` without a corpus), parsed."""
    from hitasr.core import use_dataset
    local = os.environ.get("HITASR_RECORDS_DIR")
    if local and (Path(local) / f"{name}.json").exists():
        return json.loads((Path(local) / f"{name}.json").read_text())
    if corpus is not None:
        return HitHub(spec=use_dataset(corpus, verbose=False)).load_results(name)
    path = HitHub().pull_file(f"results/{name}.json", verbose=False)
    if path is None:
        raise FileNotFoundError(f"no results/{name}.json on the Hub")
    return json.loads(Path(path).read_text())
