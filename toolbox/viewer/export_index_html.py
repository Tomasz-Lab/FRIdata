from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any
import json
import re

from tqdm import tqdm

from toolbox.config import Config, load_config


# ----------------------------
# Data models
# ----------------------------

@dataclass
class DatasetIdentity:
    db: str
    collection: str
    slug: str

    def folder_name(self) -> str:
        return f"{self.db}-{self.collection}--{self.slug}"


@dataclass
class DatasetMeta:
    identity: DatasetIdentity
    path: Path  # dataset directory containing dataset.json


@dataclass
class IndexPaths:
    forward: Optional[Path]
    reversed: Optional[Path]


# ----------------------------
# Config and roots
# ----------------------------

def resolve_datasets_root(config: Config, override_root: Optional[Path] = None) -> Path:
    base = Path(config.data_path)
    root = override_root or (base / "datasets")
    return Path(root).resolve()


# ----------------------------
# Discovery helpers
# ----------------------------

_FOLDER_NAME_RE = re.compile(
    r"^(?P<db>PDB|AFDB|ESMatlas|other)-(?P<coll>all|clust|part|subset)--(?P<slug>.+)$"
)


def _parse_identity_from_folder_name(name: str) -> Optional[DatasetIdentity]:
    m = _FOLDER_NAME_RE.match(name)
    if not m:
        return None
    return DatasetIdentity(db=m.group("db"), collection=m.group("coll"), slug=m.group("slug"))


def discover_dataset(
    dataset_arg: Optional[Path],
    datasets_root: Path,
    dataset_slug: Optional[str],
) -> DatasetMeta:
    # Case 1: dataset path provided (dir or dataset.json)
    if dataset_arg:
        dataset_arg = Path(dataset_arg).resolve()
        if dataset_arg.is_file() and dataset_arg.name == "dataset.json":
            dataset_dir = dataset_arg.parent
        elif dataset_arg.is_dir():
            dataset_dir = dataset_arg
        else:
            raise FileNotFoundError(f"Invalid dataset reference: {dataset_arg}")

        identity = _parse_identity_from_folder_name(dataset_dir.name)
        if identity is None:
            raise ValueError(
                f"Dataset folder name does not match pattern <DB>-<COLL>--<slug>: {dataset_dir.name}"
            )
        return DatasetMeta(identity=identity, path=dataset_dir)

    # Case 2: discover by slug under datasets root
    if dataset_slug:
        matches = [d for d in (datasets_root.glob("*/")) if d.is_dir() and d.name.endswith(f"--{dataset_slug}")]
        # Prefer exact enum prefix match if multiple
        for d in matches:
            identity = _parse_identity_from_folder_name(d.name)
            if identity:
                return DatasetMeta(identity=identity, path=d)

        # Fallback: user may have passed the full folder name instead of just the slug
        exact = datasets_root / dataset_slug
        if exact.is_dir():
            identity = _parse_identity_from_folder_name(exact.name)
            if identity:
                return DatasetMeta(identity=identity, path=exact)

        raise FileNotFoundError(f"Dataset with slug '{dataset_slug}' not found under {datasets_root}")

    # Case 3: nothing specified -> attempt to pick latest or raise
    raise ValueError("Either --dataset or --dataset-slug must be provided")


def find_index_files(dataset_dir: Path) -> Dict[str, IndexPaths]:
    types = ["dataset", "sequences", "coordinates", "embeddings", "distograms"]
    results: Dict[str, IndexPaths] = {}
    for t in types:
        fwd = dataset_dir / f"{t}.idx"
        rev = dataset_dir / f"{t}_reversed.idx"
        results[t] = IndexPaths(forward=fwd if fwd.exists() else None, reversed=rev if rev.exists() else None)
    return results


# ----------------------------
# Parsing helpers
# ----------------------------

def stream_parse_idx(path: Path, show_progress: bool = False, desc: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield (key, value) pairs from a top-level JSON object index file.

    This implementation loads the JSON once and streams items; suitable for moderate file sizes.
    
    Args:
        path: Path to the index file
        show_progress: If True, show a progress bar
        desc: Description for the progress bar
    """
    with path.open("r") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = list(data.items())
        iterator = tqdm(items, desc=desc, leave=False, unit="entry") if show_progress else items
        for k, v in iterator:
            yield k, v
    else:
        # Accept list of pairs [[k, v], ...] as fallback
        items = [item for item in data if isinstance(item, list) and len(item) == 2]
        iterator = tqdm(items, desc=desc, leave=False, unit="entry") if show_progress else items
        for item in iterator:
            yield item[0], item[1]


def extract_dataset_identity_from_path(path: str) -> Optional[DatasetIdentity]:
    # Pattern A: hyphenated segment
    m = re.search(
        r"(?P<db>PDB|AFDB|ESMatlas|other)-(?P<coll>all|clust|part|subset)--(?P<slug>[^/]+)",
        path,
    )
    if m:
        return DatasetIdentity(m.group("db"), m.group("coll"), m.group("slug"))

    # Pattern B: structured structures/<DB>/<COLL>_<slug>
    m2 = re.search(
        r"(?:^|/)structures/(?P<db>PDB|AFDB|ESMatlas|other)/(?P<coll>all|clust|part|subset)_(?:/)?(?P<slug>[^/]+)/",
        path,
    )
    if m2:
        return DatasetIdentity(m2.group("db"), m2.group("coll"), m2.group("slug"))

    return None


def extract_batch_id_from_path(path: str, index_type: str) -> Optional[str]:
    if index_type in ("distograms", "embeddings"):
        m = re.search(r"(?:^|/)(distograms|embeddings)/[^/]+/batch_(?P<batch>\d+)\.h5$", path)
        return m.group("batch") if m else None
    if index_type == "coordinates":
        m = re.search(r"(?:^|/)coordinates/[^/]+/batch_(?P<batch>\d+)_ca\.h5$", path)
        return m.group("batch") if m else None
    if index_type in ("dataset", "structures"):
        m = re.search(
            r"(?:^|/)structures/(PDB|AFDB|ESMatlas|other)/(all|clust|part|subset)_(?:/)?[^/]+/(?P<batch>\d+)/pdbs\.h5$",
            path,
        )
        return m.group("batch") if m else None
    if index_type == "sequences":
        return None
    return None


# ----------------------------
# Statistics
# ----------------------------

def compute_index_stats(
    index_type: str,
    forward_path: Optional[Path],
    reversed_path: Optional[Path],
    show_progress: bool = False,
) -> Dict[str, Any]:
    forward_present = bool(forward_path and forward_path.exists())
    reversed_present = bool(reversed_path and reversed_path.exists())

    num_proteins_forward = 0
    num_files_referenced = 0
    num_edges_reversed = 0

    by_dataset: Dict[str, Dict[str, Any]] = {}

    if forward_present:
        # forward maps protein_id -> file path
        try:
            desc = f"  Reading {index_type} forward index" if show_progress else ""
            for _k, _v in stream_parse_idx(forward_path, show_progress=show_progress, desc=desc):
                num_proteins_forward += 1
        except Exception:
            # Be robust if malformed
            num_proteins_forward = 0

    if reversed_present:
        seen_files = set()
        desc = f"  Reading {index_type} reversed index" if show_progress else ""
        for file_path_str, protein_ids in stream_parse_idx(reversed_path, show_progress=show_progress, desc=desc):
            seen_files.add(file_path_str)
            # protein_ids can be list or int
            if isinstance(protein_ids, list):
                count_refs = len(protein_ids)
            else:
                # If value is a single id or count
                try:
                    count_refs = int(protein_ids)
                except Exception:
                    count_refs = 1
            num_edges_reversed += count_refs

            # extract identity and batch for grouping
            identity = extract_dataset_identity_from_path(file_path_str)
            slug = identity.slug if identity else "unknown"
            ds_entry = by_dataset.setdefault(
                slug, {"files_referenced": 0, "proteins_referencing": 0, "is_self": None, "files_per_batch": {}}
            )
            ds_entry["files_referenced"] += 1
            ds_entry["proteins_referencing"] += count_refs

            batch_id = extract_batch_id_from_path(file_path_str, index_type) or "-"
            ds_entry["files_per_batch"][batch_id] = ds_entry["files_per_batch"].get(batch_id, 0) + 1

        num_files_referenced = len(seen_files)

    result = {
        "index_type": index_type,
        "forward_present": forward_present,
        "reversed_present": reversed_present,
        "num_proteins_forward": num_proteins_forward,
        "num_files_referenced": num_files_referenced,
        "num_edges_reversed": num_edges_reversed,
        "by_dataset": by_dataset,
    }
    return result


def compute_global_rollup(per_index_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Dict[str, int]] = {}
    for _t, stats in per_index_stats.items():
        for slug, entry in stats.get("by_dataset", {}).items():
            agg = aggregate.setdefault(slug, {"files_referenced": 0, "proteins_referencing": 0})
            agg["files_referenced"] += entry.get("files_referenced", 0)
            agg["proteins_referencing"] += entry.get("proteins_referencing", 0)

    # Top list by files_referenced then proteins_referencing
    top = sorted(
        (
            {
                "slug": slug,
                "files_referenced": v["files_referenced"],
                "proteins_referencing": v["proteins_referencing"],
            }
            for slug, v in aggregate.items()
        ),
        key=lambda x: (x["files_referenced"], x["proteins_referencing"]),
        reverse=True,
    )

    return {"aggregate": aggregate, "top": top[:50]}


# ----------------------------
# HTML rendering
# ----------------------------

_REPORT_JS_PATH = Path(__file__).with_name("report.js")


def _build_html(summary_payload: Dict[str, Any], report_name: str) -> str:
    data_json = json.dumps(summary_payload)
    report_js = _REPORT_JS_PATH.read_text()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{report_name}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin-right: 6px; }}
    .ok {{ background: #e8f5e9; color: #1b5e20; }}
    .no {{ background: #ffebee; color: #b71c1c; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #fafafa; }}
    tfoot td {{ border-top: 2px solid #bbb; }}
    .panel {{ border: 1px solid #eee; padding: 12px; border-radius: 6px; margin: 12px 0; }}
    details > summary {{ cursor: pointer; }}
    .muted {{ color: #666; }}
    .pct {{ float: right; color: #999; font-size: 0.9em; }}
    .controls {{ margin: 8px 0; }}
  </style>
  <script type="application/json" id="summary-data">{data_json}</script>
  <script>
{report_js}
  </script>
</head>
<body>
  <div id="root"></div>
</body>
</html>
"""


def render_html(summary_payload: Dict[str, Any], output_path: Path, report_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html(summary_payload, report_name)
    output_path.write_text(html)


# ----------------------------
# Orchestration
# ----------------------------

def export_index_view(
    config: Config,
    dataset: Optional[Path] = None,
    dataset_slug: Optional[str] = None,
    root: Optional[Path] = None,
    index_types: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    datasets_root = resolve_datasets_root(config, root)
    meta = discover_dataset(dataset, datasets_root, dataset_slug)

    index_paths = find_index_files(meta.path)
    selected_types = index_types or list(index_paths.keys())
    per_index: Dict[str, Dict[str, Any]] = {}
    
    print(f"Processing dataset: {meta.identity.folder_name()}")
    print(f"Index types to process: {', '.join(selected_types)}")
    
    for t in tqdm(selected_types, desc="Processing indexes", unit="index", position=0):
        paths = index_paths.get(t)
        if not paths:
            continue
        stats = compute_index_stats(t, paths.forward, paths.reversed, show_progress=True)
        # Mark is_self
        for slug, entry in stats.get("by_dataset", {}).items():
            entry["is_self"] = slug == meta.identity.slug
        per_index[t] = stats

    print("Computing global rollup...")
    global_rollup = compute_global_rollup(per_index)

    print("Building report payload...")
    payload = {
        "dataset": {
            "identity": {
                "db": meta.identity.db,
                "collection": meta.identity.collection,
                "slug": meta.identity.slug,
                "folder_name": meta.identity.folder_name(),
            },
            "path": str(meta.path),
        },
        "index_paths": {k: {"forward": (str(v.forward) if v.forward else None), "reversed": (str(v.reversed) if v.reversed else None)} for k, v in index_paths.items()},
        "per_index": per_index,
        "global": global_rollup,
    }

    reports_dir = (output_dir or (Path(__file__).resolve().parents[2] / "reports")).resolve()
    out_file = reports_dir / f"{meta.path.name}.html"
    
    print(f"Rendering HTML report to: {out_file}")
    render_html(payload, out_file, meta.identity.folder_name())
    
    print(f"✓ Report generated successfully: {out_file}")
    return out_file


__all__ = [
    "resolve_datasets_root",
    "discover_dataset",
    "find_index_files",
    "stream_parse_idx",
    "extract_dataset_identity_from_path",
    "extract_batch_id_from_path",
    "compute_index_stats",
    "compute_global_rollup",
    "render_html",
    "export_index_view",
]


