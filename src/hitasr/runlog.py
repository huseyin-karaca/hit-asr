"""The fold cache of the paper's notebooks: `labkit.runlog.RunLog` under `results/<corpus>/runlog/`."""

__all__ = ['RunLog', 'CacheMiss']

from labkit.runlog import CacheMiss
from labkit.runlog import RunLog as _RunLog

from hitasr.hub import HitHub


class RunLog(_RunLog):
    """`labkit.runlog.RunLog` for one corpus: the Hub is the corpus's `HitHub` (so the files go to the records
    repo), under `results/<corpus>/runlog/<name>/`. `spec=None` is the active corpus at write time."""

    def __init__(self, name, dirname=None, spec=None, repo_id=None, **kw):
        self.spec, self.repo_id = spec, repo_id
        corpus = getattr(spec, "name", None)
        super().__init__(name, dirname, root=f"results/{corpus}/runlog" if corpus else "results/runlog", **kw)

    def hub(self):
        return HitHub(self.repo_id, spec=self.spec)
