"""Hugging Face dataset repos as a cache: one listing and one parallel fetch per step, every Hub error retried
and never mistaken for a missing file, and writes that commit only what changed.

A `Hub` spans two repos: the data repo, and a records repo that receives every path under `results/` and
`studies/` (`Hub.RECORD_ROOTS`). They may be the same repo. `set_read_only()` turns every write into an error —
what a reproduction run uses, so it can never push by accident.
"""

__all__ = ['HUB_RETRIES', 'HUB_BACKOFF_S', 'HUB_BACKOFF_MAX_S', 'HUB_WORKERS', 'HUB_ETAG_TIMEOUT', 'TRANSIENT_STATUS',
           'HubUnavailable', 'HubRateLimited', 'HubReadOnly', 'hub_error_kind', 'with_retries', 'content_id',
           'is_current', 'copy_atomic', 'Fetched', 'Hub', 'set_read_only', 'read_only']

import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download

HUB_RETRIES = 7              # attempts per Hub call; the waits between them add up to ~3 min
HUB_BACKOFF_S = 5.0          # the first wait; doubled per attempt up to HUB_BACKOFF_MAX_S, +-25 % jitter
HUB_BACKOFF_MAX_S = 60.0
HUB_WORKERS = 16             # concurrent downloads in one `fetch`
HUB_ETAG_TIMEOUT = 30        # seconds for the metadata request of a download (the library's default is 10)
TRANSIENT_STATUS = (408, 425, 429, 500, 502, 503, 504)


class HubUnavailable(ConnectionError):
    """The Hub kept failing after `HUB_RETRIES` attempts. Never "absent": re-run the cell, it resumes."""


class HubRateLimited(HubUnavailable):
    """A commit hit the repo's quota (429, 128 commits an hour). Retrying within minutes is pointless."""


class HubReadOnly(RuntimeError):
    """A write was attempted while `set_read_only()` is on."""


_READ_ONLY = False


def set_read_only(flag=True):
    """Refuse every Hub write from here on (`HubReadOnly`) — or allow them again with `False`."""
    global _READ_ONLY
    _READ_ONLY = bool(flag)
    return _READ_ONLY


def read_only():
    return _READ_ONLY


def _status_code(exc):
    """The HTTP status behind `exc` or anything in its cause chain, else None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code is not None:
            return int(code)
        exc = exc.__cause__ or exc.__context__
    return None


def _brief(exc):
    code = _status_code(exc)
    return f"{type(exc).__name__}{f' {code}' if code else ''}: {(str(exc).splitlines() or [''])[0][:100]}"


def hub_error_kind(exc):
    """`"missing"` (a 404: the file or folder is not there), `"transient"` (worth retrying) or `"fatal"`.

    `huggingface_hub`'s `LocalEntryNotFoundError` subclasses `EntryNotFoundError`
    but means "the Hub could not be reached and the file is not cached" — a 504
    on a download's metadata request surfaces as exactly that — so it is
    transient, never missing.
    """
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:
        return "transient"
    if names & {"RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError"}:
        return "fatal"
    code = _status_code(exc)
    if code == 404 or (code is None and "EntryNotFoundError" in names):
        return "missing"
    if code in TRANSIENT_STATUS:
        return "transient"
    if code is not None:
        return "fatal"
    network = ("Timeout", "ConnectError", "ConnectionError", "RemoteProtocolError", "ReadError", "ChunkedEncodingError",
               "IncompleteRead", "ProtocolError")
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(k in n for n in names for k in network):
        return "transient"
    return "fatal"


def _note_retry(what, attempt, wait, exc):
    print(f"  hub: {what} failed ({_brief(exc)}); retry {attempt}/{HUB_RETRIES - 1} in {wait:.0f}s")


def with_retries(fn, what, commit=False, verbose=True, on_retry=None):
    """`fn()`, retried with exponential backoff while the Hub answers with a transient error.

    Raises `HubUnavailable` once the retries are spent, `HubRateLimited` at once
    for a 429 on a commit (`commit=True`), and anything else — a 404 included —
    as it came, so a caller can tell "absent" from "unreachable".
    """
    for attempt in range(1, HUB_RETRIES + 1):
        try:
            return fn()
        except HubUnavailable:
            raise
        except Exception as e:                                 # noqa: BLE001
            if commit and _status_code(e) == 429:
                raise HubRateLimited(f"{what}: the repo's commit quota is spent (429)") from e
            if hub_error_kind(e) != "transient":
                raise
            if attempt == HUB_RETRIES:
                raise HubUnavailable(f"{what}: the Hub failed {HUB_RETRIES} times ({_brief(e)}); re-run — "
                                     "what was already fetched or computed is kept") from e
            wait = min(HUB_BACKOFF_S * 2 ** (attempt - 1), HUB_BACKOFF_MAX_S) * random.uniform(0.75, 1.25)
            if on_retry is not None:
                on_retry(what, attempt, wait, e)
            elif verbose:
                _note_retry(what, attempt, wait, e)
            time.sleep(wait)


def content_id(path, lfs):
    """The id the Hub lists a file under: sha256 of its bytes for an LFS/Xet file, the git blob sha1 otherwise."""
    path = Path(path)
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_current(local, oid, lfs):
    """True when `local` exists and holds exactly the bytes the Hub lists as `oid`."""
    local = Path(local)
    return local.is_file() and content_id(local, lfs) == oid


def copy_atomic(src, dst):
    """Copy `src` to `dst` through a hidden temporary next to it, so a half-written file never looks complete."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}-{threading.get_ident()}.hubpart")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


@dataclass
class Fetched:
    """What one `HitHub.fetch` did: counts, and the repo paths the Hub does not have."""
    wanted: int = 0
    current: int = 0
    fetched: int = 0
    seconds: float = 0.0
    absent: list = field(default_factory=list)

    def line(self, what):
        on_hub = self.wanted - len(self.absent)
        took = f", fetched {self.fetched} in {self.seconds:.0f}s" if self.fetched else ""
        return f"  hub: {what} — {on_hub} of {self.wanted} on the Hub, {self.current} already local{took}"


class Hub:
    """Read and write a data repo and a records repo on the Hugging Face Hub, as a cache.

    Parameters
    ----------
    repo_id : the data repo (a dataset repo).
    records_repo : where every path under `RECORD_ROOTS` goes; `None` means `repo_id`.
    """

    RECORD_ROOTS = ("results/", "studies/")

    def __init__(self, repo_id, records_repo=None):
        self.repo_id = repo_id
        self.records_repo = records_repo or repo_id
        self.api = HfApi()
        self._files = None                      # cached; invalidated by every push
        self._filesystem = None
        self._revision = None

    def __repr__(self):
        rec = f", records {self.records_repo!r}" if self.records_repo != self.repo_id else ""
        return f"{type(self).__name__}({self.repo_id!r}{rec})"

    def repo_for(self, path):
        """The repo `path` lives in: the records repo under `RECORD_ROOTS`, the data repo otherwise."""
        return self.records_repo if str(path).lstrip("/").startswith(self.RECORD_ROOTS) else self.repo_id

    def repos(self):
        return list(dict.fromkeys((self.repo_id, self.records_repo)))

    def _writable(self, what):
        if _READ_ONLY:
            raise HubReadOnly(f"{what}: the Hub is read-only in this session (labkit.hub.set_read_only)")

    def revision(self):
        """The data repo's current commit sha, or `None` if the Hub is unreachable.

        The cache key everything downstream hangs off: re-pushing an artefact
        changes the sha, so a stale local cache is invalidated rather than
        served silently. `None` means "could not verify".
        """
        if self._revision is None:
            try:
                self._revision = self.api.dataset_info(self.repo_id).sha
            except Exception:                              # noqa: BLE001 — offline
                return None
        return self._revision

    def files(self, refresh=False):
        """Every file path in the data repo and the records repo. Cached after the first call.

        The whole repo — thousands of paths, several paged requests. A cache
        that needs one folder uses `listing(prefix)` instead.
        """
        if refresh or self._files is None:
            self._files = [f for repo in self.repos()
                           for f in with_retries(lambda repo=repo: self.api.list_repo_files(repo, repo_type="dataset"),
                                                 f"list {repo}")]
        return self._files

    def local_path(self, path):
        """Download one repo file into the `huggingface_hub` cache (retried); return its local path."""
        return with_retries(lambda: self._download(path), f"download {path}")

    def prefetch(self, patterns, verbose=True, strict=False):
        """Download every file matching `patterns` in ONE parallel pass. Returns the snapshot dir.

        One `snapshot_download` beats hundreds of serialised `hf_hub_download`s
        by an order of magnitude on a high-latency link, and the bytes land in
        the same cache, so the second session pays nothing. Retried as a whole
        (finished files stay in the cache and a partial one resumes); once the
        retries are spent it returns `None` — or, with `strict`, raises.
        """
        repos = {self.repo_for(p) for p in patterns}
        if len(repos) != 1:
            raise ValueError(f"prefetch patterns span {sorted(repos)}; call once per repo")
        repo = repos.pop()
        try:
            return with_retries(lambda: snapshot_download(repo, repo_type="dataset",
                                                          allow_patterns=list(patterns), etag_timeout=HUB_ETAG_TIMEOUT),
                                f"snapshot {', '.join(patterns)[:80]}", verbose=verbose)
        except Exception as e:                                 # noqa: BLE001
            if strict:
                raise
            if verbose:
                print(f"  prefetch skipped ({type(e).__name__}: {e}); reading per file")
            return None

    def _fs(self):
        if self._filesystem is None:
            self._filesystem = HfFileSystem()
        return self._filesystem

    def _tree(self, prefix):
        """`[(path, content id, is_lfs)]` of every file under the folder `prefix` — one paged request."""
        from huggingface_hub.hf_api import RepoFile

        out = []
        for e in self.api.list_repo_tree(self.repo_for(prefix), repo_type="dataset", path_in_repo=prefix.rstrip("/") or None,
                                         recursive=True):
            if isinstance(e, RepoFile):
                sha = getattr(e.lfs, "sha256", None) or (e.lfs.get("sha256") if isinstance(e.lfs, dict) else None)
                out.append((e.path, sha or e.blob_id, sha is not None))
        return out

    def _paths_info(self, paths):
        """`[(path, content id, is_lfs)]` for those of `paths` that exist — one request."""
        out = []
        for repo in self.repos():
            mine = [p for p in paths if self.repo_for(p) == repo]
            if not mine:
                continue
            for e in self.api.get_paths_info(repo, mine, repo_type="dataset"):
                if hasattr(e, "blob_id"):
                    sha = getattr(e.lfs, "sha256", None) or (e.lfs.get("sha256") if isinstance(e.lfs, dict) else None)
                    out.append((e.path, sha or e.blob_id, sha is not None))
        return out

    def _download(self, path):
        """One file into the `huggingface_hub` cache; its local path."""
        return hf_hub_download(self.repo_for(path), filename=path, repo_type="dataset", etag_timeout=HUB_ETAG_TIMEOUT)

    def _commit(self, files, message):
        """`{path_in_repo: local_path}` as ONE commit per repo."""
        from huggingface_hub import CommitOperationAdd

        for repo in self.repos():
            ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local)) for p, local in files.items()
                   if self.repo_for(p) == repo]
            if ops:
                self.api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message=message)

    def listing(self, prefix):
        """`{path: (content id, is_lfs)}` of every file under the folder `prefix`; `{}` when there is no such folder.

        One request (retried), always fresh. This is how a cache learns what
        the Hub has — and a file it lists is one a failed download must not be
        mistaken for missing.
        """
        try:
            tree = with_retries(lambda: self._tree(prefix), f"list {prefix}")
        except Exception as e:                                 # noqa: BLE001
            if not isinstance(e, HubUnavailable) and hub_error_kind(e) == "missing":
                return {}
            raise
        return {p: (oid, lfs) for p, oid, lfs in tree}

    def remote_ids(self, paths):
        """`{path: (content id, is_lfs)}` for those of `paths` the Hub has (retried; one request)."""
        if not paths:
            return {}
        return {p: (oid, lfs) for p, oid, lfs in with_retries(lambda: self._paths_info(paths),
                                                               f"look up {len(paths)} path(s)")}

    def fetch(self, targets, listing, what="files", verbose=True):
        """Bring every `{repo_path: local_path}` the `listing` has up to date locally. Returns a `Fetched`.

        A local copy whose content id equals the listing's costs no request;
        the rest are downloaded concurrently (`HUB_WORKERS`), each retried, and
        written atomically. A path the listing lacks is reported in `absent`
        and never requested. If a download still fails, the queue is abandoned
        and `HubUnavailable` is raised naming it — the files already fetched
        stay, so a re-run resumes where this stopped.
        """
        t0 = time.perf_counter()
        report = Fetched(wanted=len(targets))
        todo = {}
        for repo, local in targets.items():
            if repo not in listing:
                report.absent.append(repo)
            elif is_current(local, *listing[repo]):
                report.current += 1
            else:
                todo[repo] = Path(local)
        done, failed, gone = [], [], []
        stop, lock, last = threading.Event(), threading.Lock(), [0.0]

        def note(w, attempt, wait, exc):                   # one line per 20 s, not one per thread
            with lock:
                if verbose and time.monotonic() - last[0] > 20:
                    last[0] = time.monotonic()
                    _note_retry(f"{what} ({len(todo)} file(s))", attempt, wait, exc)

        def one(item):
            repo, local = item
            if stop.is_set():
                return
            try:
                copy_atomic(with_retries(lambda: self._download(repo), f"download {repo}", on_retry=note), local)
                with lock:
                    done.append(repo)
            except Exception as e:                             # noqa: BLE001
                with lock:
                    if hub_error_kind(e) == "missing":        # deleted between the listing and the download
                        gone.append(repo)
                    else:
                        failed.append((repo, e))
                        stop.set()

        if todo:
            with ThreadPoolExecutor(max_workers=min(HUB_WORKERS, len(todo))) as pool:
                list(pool.map(one, todo.items()))
        report.absent += gone
        report.fetched, report.seconds = len(done), time.perf_counter() - t0
        if failed:
            repo, e = failed[0]
            left = len(todo) - len(done) - len(gone)
            raise HubUnavailable(f"{what}: {repo} could not be downloaded ({_brief(e)}); {left} of {len(todo)} file(s) "
                                 "not fetched. Nothing was computed in their place — re-run, the fetched ones are kept"
                                 ) from e
        return report

    def push_json(self, payload, path_in_repo, message=None, skip_identical=True):
        """Upload a dict as a JSON file inside the repo (`results/*.json`). Identical bytes make no commit."""
        local = Path(path_in_repo.replace("/", "_"))
        local.write_text(json.dumps(payload, indent=2, default=str))
        if self.push_files({path_in_repo: local}, message or f"Update {path_in_repo}", verbose=False,
                           skip_identical=skip_identical):
            print(f"pushed file    {path_in_repo}")
        else:
            print(f"unchanged      {path_in_repo} (identical on the Hub; no commit)")

    def push_file(self, local_path, path_in_repo, message=None, verbose=True, skip_identical=False):
        """Upload one local file verbatim (retried). Returns `path_in_repo`, or `None` when skipped as identical."""
        pushed = self.push_files({path_in_repo: local_path}, message or f"Update {path_in_repo}", verbose=False,
                                 skip_identical=skip_identical)
        if verbose and pushed:
            size = Path(local_path).stat().st_size
            print(f"pushed file    {path_in_repo}  ({size / 1e6:.1f} MB)")
        return path_in_repo if pushed else None

    def push_files(self, files, message=None, verbose=True, skip_identical=False):
        """Upload several local files in ONE commit. `files` is `{path_in_repo: local_path}`. Returns the paths sent.

        Retried on 5xx and timeouts (a 504 on a commit often means it went
        through; the retry then changes nothing); a 429 raises `HubRateLimited`
        at once. `skip_identical` first asks the Hub for the files' content ids
        and leaves out every file it already holds byte for byte — a commit
        that changes nothing still spends the hourly quota and moves the head.
        """
        files = {repo: Path(local) for repo, local in files.items()}
        if files:
            self._writable(f"commit {len(files)} file(s)")
        if skip_identical and files:
            have = self.remote_ids(list(files))
            files = {r: l for r, l in files.items() if not (r in have and is_current(l, *have[r]))}
        if not files:
            return []
        message = message or f"Update {len(files)} file(s)"
        with_retries(lambda: self._commit(files, message), f"commit {len(files)} file(s)", commit=True)
        self._files = None
        if verbose:
            size = sum(v.stat().st_size for v in files.values())
            print(f"pushed {len(files)} file(s) in one commit  ({size / 1e6:.1f} MB)")
        return list(files)

    def push_folder(self, local_dir, path_in_repo, message=None, verbose=True):
        """Upload a directory tree as `path_in_repo/` in one commit.

        `upload_folder` streams each file through LFS and resumes a shard that
        was already uploaded, which with 2 GB shards is the whole robustness
        story; `ASRFrameExtractor.push` calls it once per split so no single
        commit carries more than one split's dozen shards.
        """
        self._writable(f"upload {path_in_repo}/")
        with_retries(lambda: self.api.upload_folder(repo_id=self.repo_for(path_in_repo), repo_type="dataset",
                                                    folder_path=str(local_dir), path_in_repo=path_in_repo,
                                                    commit_message=message or f"Update {path_in_repo}/"),
                     f"upload {path_in_repo}/", commit=True)
        self._files = None
        if verbose:
            n = sum(1 for p in Path(local_dir).rglob("*") if p.is_file())
            size = sum(p.stat().st_size for p in Path(local_dir).rglob("*") if p.is_file())
            print(f"pushed folder  {path_in_repo}/  ({n} file(s), {size / 1e9:.2f} GB)")

    def pull_file(self, path_in_repo, local_path=None, verbose=True):
        """Download one file, or return `None` if the Hub does not have it.

        `None` rather than an exception because a resumable study cannot tell
        "first run" from "lost the database" any other way — and ONLY for a
        404: a Hub that does not answer is retried and then raises
        `HubUnavailable`, never passes for absence. With `local_path` the file
        is COPIED out of the cache, because the cache entry is replaced on the
        next download and an SQLite file opened from it would change under
        the connection.
        """
        try:
            cached = with_retries(lambda: self._download(path_in_repo), f"download {path_in_repo}", verbose=verbose)
        except Exception as e:                                 # noqa: BLE001
            if isinstance(e, HubUnavailable) or hub_error_kind(e) != "missing":
                raise
            if verbose:
                print(f"  no {path_in_repo} in {self.repo_for(path_in_repo)}")
            return None
        if local_path is None:
            return cached
        local_path = copy_atomic(cached, local_path)
        if verbose:
            print(f"pulled file    {path_in_repo} -> {local_path} "
                  f"({local_path.stat().st_size / 1e6:.1f} MB)")
        return str(local_path)
