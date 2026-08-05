"""Bounded schema probes that never confirm inferred battery semantics."""

import csv
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from scipy.io import whosmat

from tristatelite.data.archive_inventory import inventory_zip

TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".mat"}
MAX_TEXT_BYTES = 64 * 1024
MAX_ROWS = 20
ROLE_KEYWORDS = {
    "time": {"time", "timestamp", "seconds", "sec", "t"},
    "voltage": {"voltage", "volt", "v"},
    "current": {"current", "amp", "amps", "a", "i"},
    "temperature": {"temperature", "temp", "celsius", "degc"},
    "cycle": {"cycle", "cycle_id", "cycleindex"},
    "battery": {"battery", "pack", "cell", "battery_id", "cell_id"},
    "phase": {"phase", "mode", "type", "state", "operation"},
    "capacity": {"capacity", "cap", "ah", "amp_hour"},
}
HUMAN_DECISIONS = (
    "battery_id source",
    "cycle_id source or cycle-boundary rule",
    "time field and unit",
    "voltage field and unit",
    "current field, unit, and sign convention",
    "temperature field and unit, or confirmation that it is absent",
    "phase field or phase-inference rule",
    "capacity field or confirmation that it must be integrated",
    "complete-discharge termination rule",
)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _candidate_roles(names: list[str]) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for name in names:
        normalized = _normalize_name(name)
        tokens = set(normalized.split("_"))
        for role, keywords in ROLE_KEYWORDS.items():
            matches = sorted(
                keyword
                for keyword in keywords
                if normalized == keyword or (len(keyword) > 1 and keyword in tokens)
            )
            if not matches:
                continue
            result[role].append(
                {
                    "name": name,
                    "score": float(len(matches)),
                    "evidence": [
                        f"keyword {keyword!r} matched normalized name {normalized!r}"
                        for keyword in matches
                    ],
                    "status": "unconfirmed",
                }
            )
    return {role: candidates for role, candidates in sorted(result.items())}


def _decode_text(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding is total for byte strings")


def _infer_delimiter(text: str, suffix: str) -> str | None:
    sample = "\n".join(text.splitlines()[:MAX_ROWS])
    if not sample:
        return None
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t; ").delimiter
    except csv.Error:
        defaults = {".csv": ",", ".tsv": "\t"}
        return defaults.get(suffix)


def _probe_text(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, object]:
    with archive.open(info) as stream:
        raw = stream.read(MAX_TEXT_BYTES)
    text, encoding = _decode_text(raw)
    suffix = PurePosixPath(info.filename).suffix.lower()
    delimiter = _infer_delimiter(text, suffix)
    lines = text.splitlines()[:MAX_ROWS]
    if delimiter == " ":
        rows = [line.split() for line in lines if line.strip()]
        delimiter_label = "whitespace"
    elif delimiter:
        rows = list(csv.reader(lines, delimiter=delimiter))
        delimiter_label = delimiter
    else:
        rows = [[line] for line in lines if line.strip()]
        delimiter_label = None
    headers = [value.strip() for value in rows[0]] if rows else []
    sample_rows = [[str(value) for value in row] for row in rows[1:6]]
    widths = {len(row) for row in rows}
    return {
        "path": info.filename,
        "suffix": suffix,
        "probe_type": "text",
        "probe_status": "probed",
        "encoding_candidate": encoding,
        "delimiter": delimiter_label,
        "headers": headers,
        "sample_rows": sample_rows,
        "row_width_consistent": len(widths) <= 1,
        "candidate_roles": _candidate_roles(headers),
    }


def _probe_mat(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="tristatelite-mat-") as temp_dir:
        extracted = Path(temp_dir) / PurePosixPath(info.filename).name
        with archive.open(info) as source, extracted.open("wb") as target:
            shutil.copyfileobj(source, target)
        try:
            variables = [
                {"name": name, "shape": list(shape), "matlab_class": matlab_class}
                for name, shape, matlab_class in whosmat(extracted)
            ]
        except (OSError, TypeError, ValueError) as exc:
            return {
                "path": info.filename,
                "suffix": ".mat",
                "probe_type": "matlab",
                "probe_status": "hdf5_mat_requires_phase2_support",
                "error": str(exc),
                "mat_variables": [],
                "candidate_roles": {},
            }
    return {
        "path": info.filename,
        "suffix": ".mat",
        "probe_type": "matlab",
        "probe_status": "probed",
        "mat_variables": variables,
        "candidate_roles": _candidate_roles([item["name"] for item in variables]),
    }


def probe_member(archive: Path, member_path: str) -> dict[str, object]:
    """Probe one supported, safe ZIP member using bounded metadata reads."""
    with zipfile.ZipFile(archive) as zip_archive:
        info = zip_archive.getinfo(member_path)
        normalized = PurePosixPath(info.filename.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            return {"path": info.filename, "probe_status": "skipped_unsafe_path"}
        if info.flag_bits & 0x1:
            return {"path": info.filename, "probe_status": "skipped_encrypted"}
        suffix = normalized.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            return _probe_text(zip_archive, info)
        if suffix == ".mat":
            return _probe_mat(zip_archive, info)
        return {"path": info.filename, "suffix": suffix, "probe_status": "unsupported"}


def _is_documentation(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return "readme" in name


def _selected_probe_paths(members: list[dict[str, object]]) -> list[str]:
    supported = [member for member in members if member["suffix"] in SUPPORTED_SUFFIXES]
    if len(supported) <= 200:
        return [str(member["path"]) for member in supported]
    per_extension: dict[str, list[str]] = defaultdict(list)
    for member in supported:
        per_extension[str(member["suffix"])].append(str(member["path"]))
    selected = {path for paths in per_extension.values() for path in paths[:20]}
    selected.update(
        str(member["path"]) for member in members if _is_documentation(str(member["path"]))
    )
    return sorted(selected)


def build_schema_report(archive: Path) -> dict[str, object]:
    """Inventory an archive and aggregate unconfirmed schema candidates."""
    inventory = inventory_zip(archive)
    members = list(inventory["members"])
    skipped = sorted(
        str(member["path"]) for member in members if member["unsafe_path"] or member["is_encrypted"]
    )
    by_path = {str(member["path"]): member for member in members}
    selected = _selected_probe_paths(members)
    probes = [
        probe_member(archive, path) for path in selected if path not in skipped and path in by_path
    ]

    ranked = []
    required_roles = {"time", "voltage", "current"}
    for probe in probes:
        candidates = probe.get("candidate_roles", {})
        present = required_roles.intersection(candidates)
        role_score = sum(max(float(item["score"]) for item in candidates[role]) for role in present)
        if present:
            ranked.append(
                {
                    "path": probe["path"],
                    "probe_type": probe["probe_type"],
                    "required_role_count": len(present),
                    "required_roles": sorted(present),
                    "score": len(present) * 10 + role_score,
                }
            )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["path"])))

    semantic_candidates = []
    for probe in probes:
        for role, candidates in probe.get("candidate_roles", {}).items():
            for candidate in candidates:
                semantic_candidates.append({"path": probe["path"], "role": role, **candidate})
    semantic_candidates.sort(key=lambda item: (str(item["role"]), str(item["path"])))

    patterns = Counter(
        re.sub(r"\d+", "<N>", PurePosixPath(str(member["path"])).name.lower()) for member in members
    )
    unsupported = {
        suffix: count
        for suffix, count in inventory["extensions"].items()
        if suffix not in SUPPORTED_SUFFIXES
    }
    return {
        "archive_path": inventory["archive_path"],
        "archive_bytes": inventory["archive_bytes"],
        "archive_sha256": inventory["archive_sha256"],
        "inventory_member_count": inventory["member_count"],
        "documentation_files": sorted(
            str(member["path"]) for member in members if _is_documentation(str(member["path"]))
        ),
        "probes": probes,
        "candidate_timeseries_files": ranked,
        "semantic_candidates": semantic_candidates,
        "path_groups": inventory["top_level"],
        "filename_patterns": dict(sorted(patterns.items())),
        "unsupported_extension_counts": dict(sorted(unsupported.items())),
        "skipped_unsafe_or_encrypted": skipped,
    }


def validate_report_consistency(
    inventory: dict[str, object],
    schema: dict[str, object],
    download_manifest: dict[str, object] | None = None,
) -> None:
    """Reject inconsistent reports or accidentally confirmed candidates."""
    inventory_hash = inventory["archive_sha256"]
    if schema["archive_sha256"] != inventory_hash:
        raise ValueError("schema report SHA256 does not match archive inventory SHA256")
    if download_manifest is not None and download_manifest.get("sha256") != inventory_hash:
        raise ValueError("download manifest SHA256 does not match archive inventory SHA256")

    members = {str(member["path"]): member for member in inventory["members"]}
    probed_paths = {str(probe["path"]) for probe in schema["probes"]}
    if not probed_paths.issubset(members):
        raise ValueError("schema report contains a probe absent from the archive inventory")
    prohibited = {
        path
        for path, member in members.items()
        if member["unsafe_path"] or member["is_encrypted"]
    }
    if probed_paths.intersection(prohibited):
        raise ValueError("unsafe or encrypted archive members must not be probed")

    candidates = list(schema["semantic_candidates"])
    candidates.extend(
        candidate
        for probe in schema["probes"]
        for role_candidates in probe.get("candidate_roles", {}).values()
        for candidate in role_candidates
    )
    if any(candidate.get("status") != "unconfirmed" for candidate in candidates):
        raise ValueError("every candidate semantic role must remain unconfirmed")


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _count_table(values: dict[str, int], label: str) -> list[str]:
    lines = [f"| {label} | Count |", "|---|---:|"]
    lines.extend(f"| {_markdown_cell(key or '(no suffix)')} | {count} |" for key, count in values.items())
    if not values:
        lines.append("| (none) | 0 |")
    return lines


def render_schema_markdown(
    inventory: dict[str, object], schema: dict[str, object]
) -> str:
    """Render the required human-review report without confirming semantics."""
    lines = [
        "# NASA Archive Schema Discovery Report",
        "",
        "## 1. Archive provenance",
        "",
        f"- Path: `{inventory['archive_path']}`",
        f"- Bytes: {inventory['archive_bytes']}",
        f"- SHA256: `{inventory['archive_sha256']}`",
        "",
        "## 2. Extension counts",
        "",
        *_count_table(inventory["extensions"], "Extension"),
        "",
        "## 3. Top-level directory counts",
        "",
        *_count_table(inventory["top_level"], "Top-level token"),
        "",
        "## 4. Candidate README/documentation files",
        "",
    ]
    documentation = schema["documentation_files"]
    lines.extend(f"- `{path}`" for path in documentation)
    if not documentation:
        lines.append("- None detected")

    lines.extend(
        [
            "",
            "## 5. Candidate time-series files",
            "",
            "| Rank | Path | Required roles found | Score |",
            "|---:|---|---|---:|",
        ]
    )
    for rank, candidate in enumerate(schema["candidate_timeseries_files"], start=1):
        roles = ", ".join(candidate["required_roles"])
        lines.append(
            f"| {rank} | `{_markdown_cell(candidate['path'])}` | {roles} | "
            f"{candidate['score']} |"
        )
    if not schema["candidate_timeseries_files"]:
        lines.append("| - | None detected | - | - |")

    lines.extend(
        [
            "",
            "## 6. Candidate semantic fields",
            "",
            "All roles below are keyword-only candidates with status `unconfirmed`.",
            "",
            "| Path | Role | Field/variable | Score | Status | Evidence |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for candidate in schema["semantic_candidates"]:
        evidence = "; ".join(candidate["evidence"])
        lines.append(
            f"| `{_markdown_cell(candidate['path'])}` | {candidate['role']} | "
            f"`{_markdown_cell(candidate['name'])}` | {candidate['score']} | "
            f"{candidate['status']} | {_markdown_cell(evidence)} |"
        )
    if not schema["semantic_candidates"]:
        lines.append("| - | - | - | - | unconfirmed | No keyword candidates detected |")

    lines.extend(
        [
            "",
            "## 7. MATLAB variable inventories",
            "",
            "| Path | Probe status | Variable | Shape | MATLAB class |",
            "|---|---|---|---|---|",
        ]
    )
    mat_rows = 0
    for probe in schema["probes"]:
        if probe.get("probe_type") != "matlab":
            continue
        variables = probe.get("mat_variables", [])
        if not variables:
            lines.append(
                f"| `{_markdown_cell(probe['path'])}` | {probe['probe_status']} | - | - | - |"
            )
            mat_rows += 1
        for variable in variables:
            lines.append(
                f"| `{_markdown_cell(probe['path'])}` | {probe['probe_status']} | "
                f"`{_markdown_cell(variable['name'])}` | `{variable['shape']}` | "
                f"{variable['matlab_class']} |"
            )
            mat_rows += 1
    if not mat_rows:
        lines.append("| - | No MATLAB members probed | - | - | - |")

    lines.extend(
        [
            "",
            "## 8. Unsafe/encrypted/unsupported members",
            "",
        ]
    )
    skipped = schema["skipped_unsafe_or_encrypted"]
    lines.extend(f"- Skipped unsafe/encrypted: `{path}`" for path in skipped)
    if not skipped:
        lines.append("- Unsafe/encrypted members: none detected")
    unsupported = schema["unsupported_extension_counts"]
    lines.extend(f"- Unsupported `{suffix or '(no suffix)'}`: {count}" for suffix, count in unsupported.items())
    if not unsupported:
        lines.append("- Unsupported extensions: none detected")

    lines.extend(
        [
            "",
            "## 9. Explicit human decisions required",
            "",
            "| Decision | Status |",
            "|---|---|",
            *(f"| {decision} | unresolved |" for decision in HUMAN_DECISIONS),
            "",
            "## 10. Phase 1 stop",
            "",
            (
                "Phase 1 stop: review and explicitly confirm the field mapping and cycle semantics "
                "before building labels, splits, windows, models, or training."
            ),
            "",
        ]
    )
    return "\n".join(lines)
