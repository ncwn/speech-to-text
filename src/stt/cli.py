"""Command-line interface.

stt backends                     what is installed and working
stt hardware                     what this machine is and its fastest precision
stt models                       available model names per backend
stt fetch-fleurs                 download Burmese eval audio + references
stt transcribe AUDIO...          run one backend
stt eval RESULTS.jsonl           score a run against references
stt align AUDIO --text FILE      time an existing transcript (subtitles)
stt compare AUDIO...             run every installed backend and compare
stt vote RUNS...                 combine runs by per-character vote
stt route BASE STRONG            re-transcribe only the least-confident spans
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from stt import audio as audio_mod
from stt import bench as bench_mod
from stt.burmese import NormalizeOptions, describe_encoding
from stt.cascade import DEFAULT_BLOCK, DEFAULT_ESCALATE
from stt.evaluate import corpus_rtf, load_references, score_results
from stt.evidence import EvidenceError, check_evidence, update_evidence
from stt.execution import mark_run_trust, transcribe_corpus
from stt.measurement import (
    AudioInput,
    SubjectSpec,
    WorkerRequest,
    new_run_id,
    write_json_atomic,
)
from stt.provenance import ProvenanceError, preflight_model_binding, validate_binding
from stt.registry import all_backends, get_backend
from stt.results import (
    TranscriptionResult,
    read_jsonl,
    write_jsonl,
    write_srt,
    write_text,
    write_vtt,
)
from stt.telemetry import ResourceUsage, describe_host, measure, saturation
from stt.vote import DEFAULT_WEIGHTS, VoteInputError, prepare_vote_groups, rover

app = typer.Typer(
    name="stt",
    help="Burmese-focused speech-to-text testing harness.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

#: omniASR's code for Burmese.
BURMESE = "mya_Mymr"

# Voting and routing combine already-executed models, but no complete derived
# model-lineage manifest exists yet. Keep those artifacts diagnostic until one
# does, rather than copying a single constituent's provenance onto the result.
_DERIVED_VOTE_TRUST_ISSUE = "derived vote lacks complete model provenance"
_DERIVED_ROUTE_TRUST_ISSUE = "derived route lacks complete model provenance"

DEFAULT_CACHE = Path("data/.converted")
DEFAULT_OUTPUT_DIR = Path("outputs")


def _fmt(value: float | None, spec: str = ".4f", dash: str = "—") -> str:
    if value is None or value != value:  # None or NaN
        return dash
    return format(value, spec)


@app.command()
def backends() -> None:
    """List backends and whether their runtime is installed."""
    table = Table(title="ASR backends")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Description")

    for name, cls in all_backends().items():
        ok, reason = cls.is_available()
        status = f"[green]ready[/green] ({reason})" if ok else f"[red]missing[/red] — {reason}"
        if not ok and cls.install_hint:
            status += f"\n[dim]install: {cls.install_hint}[/dim]"
        table.add_row(name, status, cls.description)

    console.print(table)


@app.command()
def models(
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="Show models for one backend only")
    ] = None,
) -> None:
    """List the model names each backend accepts for ``-m``."""
    from stt.backends import dolphin, omniasr_gguf, omniasr_torch, transformers_asr

    def size(spec: Any) -> str:
        # Decimal GB: these are the vendors' own published figures, and every
        # doc quotes them that way (6500 MB is "6.5 GB", not 6.3).
        mb = spec.approx_mb
        return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb} MB"

    def length(spec: Any) -> str:
        return "unlimited" if spec.unlimited else "40 s max"

    # (backend, device, extra columns as (heading, justify, value-fn), footnote)
    catalogue = [
        (
            omniasr_torch,
            "omniasr-torch  (Metal)",
            [("Download", "right", size), ("Long audio", "left", length)],
            "Any card from facebookresearch/omnilingual-asr works; these are the common ones.",
        ),
        (
            omniasr_gguf,
            "omniasr-gguf  (Metal)",
            [("Size", "right", size), ("Long audio", "left", length)],
            "4-bit cards are for iteration only — see docs/findings.md#quantisation.",
        ),
        (
            transformers_asr,
            "hf  (Metal via transformers)",
            [
                ("Download", "right", size),
                ("Family", "left", lambda s: s.family),
                ("Notes", "left", lambda s: s.note),
            ],
            "MMS and SeamlessM4T weights are CC-BY-NC-4.0.",
        ),
        (
            dolphin,
            "dolphin  (CPU)",
            [("Params", "right", lambda s: f"{s.params_m}M"), ("Download", "right", size)],
            "Only base and small were publicly released.",
        ),
    ]

    for module, title, columns, footnote in catalogue:
        name = title.split()[0]
        if backend not in (None, name):
            continue
        table = Table(title=title)
        table.add_column("Model", style="bold")
        for heading, justify, _ in columns:
            table.add_column(heading, justify=justify)
        for key, spec in module.MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == module.DEFAULT_MODEL else key
            table.add_row(label, *(fn(spec) for _, _, fn in columns))
        console.print(table)
        console.print(f"[dim]{footnote}[/dim]")


@app.command()
def hardware(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-run the dtype probe instead of using the cache")
    ] = False,
) -> None:
    """Show what this machine is, and which precision it runs fastest.

    Everything shown is read or measured from the machine — nothing about the
    chip is written down in this repo.
    """
    from stt.hardware import describe, fastest_dtype

    facts = describe()
    table = Table(title=facts["chip"])
    table.add_column("", style="bold")
    table.add_column("")
    for name, count in facts["cores"].items():
        table.add_row(f"{name} cores", str(count))
    table.add_row("RAM", f"{facts['ram_mb'] / 1024:.0f} GB" if facts["ram_mb"] else "—")
    table.add_row("platform", facts["platform"])

    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        table.add_row("torch device", device)
        threads = f"{torch.get_num_threads()}  [dim](its own default)[/dim]"
        table.add_row("torch threads", threads)
        with console.status("Timing float16 against bfloat16…" if refresh else "Reading probe…"):
            best = fastest_dtype(device, refresh=refresh)
        table.add_row("fastest GPU dtype", best)
    except ImportError:
        table.add_row("torch", "[dim]not installed[/dim]")

    console.print(table)
    console.print(
        "[dim]Thread counts are left to macOS and to each runtime — "
        "see docs/findings.md#threads.[/dim]"
    )


@app.command("fetch-fleurs")
def fetch_fleurs_cmd(
    dest: Annotated[Path, typer.Option("--dest", help="Where to put audio and references")] = Path(
        "data/fleurs"
    ),
    split: Annotated[str, typer.Option(help="dev, test, or train")] = "dev",
    limit: Annotated[int, typer.Option(help="Number of clips; 0 for the whole split")] = 20,
    language: Annotated[str, typer.Option(help="FLEURS config code")] = "my_mm",
) -> None:
    """Download Burmese audio with reference transcripts from FLEURS."""
    from stt.datasets import fetch_fleurs

    console.print(f"Fetching FLEURS [bold]{language}[/bold] {split} split…")
    audio_dir, refs = fetch_fleurs(dest, split=split, limit=limit or None, config=language)
    n = len(list(audio_dir.glob("*.wav")))
    console.print(f"[green]{n} clips[/green] → {audio_dir}")
    console.print(f"[green]references[/green] → {refs}")

    sample = load_references(refs)
    if sample:
        first = next(iter(sample.values()))
        console.print(f"Reference encoding: [bold]{describe_encoding(first)}[/bold]")


def _report_resources(results: list[TranscriptionResult]) -> None:
    """Summarise what the run cost, beyond wall-clock time."""
    from stt.hardware import compute_threads

    usages = _unique_usages(results)
    if not usages:
        return

    aggregate = _aggregate_usage(usages)
    line = (
        f"[dim]CPU {aggregate.cpu_utilization:.1f} cores busy"
        if aggregate.cpu_utilization is not None
        else "[dim]CPU —"
    )
    if aggregate.gpu_util is not None:
        series = aggregate.gpu_util
        line += (
            f" · GPU p50 {series.p50:.0f}% (mean {series.mean:.0f}%, "
            f"peak {series.max:.0f}%, n={series.n}, idle {series.idle_pct:.0f}%)"
        )
    elif aggregate.gpu_util_mean is not None:
        line += f" · GPU mean {aggregate.gpu_util_mean:.0f}% (legacy; n unknown)"
    if aggregate.gpu_mb is not None:
        line += f", {aggregate.gpu_mb:.0f} MB"
    if aggregate.rss_peak_mb is not None:
        line += f" · current RSS peak {aggregate.rss_peak_mb:.0f} MB"
    else:
        line += f" · process-lifetime RSS high-water {aggregate.peak_rss_mb:.0f} MB"
    console.print(line + "[/dim]")

    # Which resource to blame, which RTF alone cannot say.
    console.print(f"[dim]→ {saturation(aggregate, compute_threads())}[/dim]")


def _unique_usages(results: list[TranscriptionResult]) -> list[ResourceUsage]:
    """Return one resource window per file or batch, including failed work."""
    usages: list[ResourceUsage] = []
    seen_batches: set[object] = set()
    for result in results:
        if result.resources is None:
            continue
        if result.metadata.get("resource_scope") == "batch":
            batch_id = result.metadata.get("batch_id")
            if batch_id in seen_batches:
                continue
            seen_batches.add(batch_id)
        usages.append(result.resources)
    return usages


def _common_wall_rtf(
    results: list[TranscriptionResult], *, allow_incomplete: bool = False
) -> float | None:
    """Use the shared outer corpus wall when current results provide one."""
    if not results or (
        not allow_incomplete and any(result.error or not result.trusted for result in results)
    ):
        return None
    durations = [
        result.audio_duration_s
        for result in results
        if result.audio_duration_s is not None and result.audio_duration_s > 0
    ]
    if len(durations) != len(results):
        return None
    corpus_windows = [
        result.resources
        for result in results
        if result.resources is not None and result.metadata.get("resource_scope") == "corpus"
    ]
    if corpus_windows:
        return sum(usage.wall_s for usage in corpus_windows) / sum(durations)
    return corpus_rtf(results)


def _aggregate_usage(usages: list[ResourceUsage]) -> ResourceUsage:
    """Pool resource windows without re-averaging per-file rates."""
    from stt.telemetry import pool_series

    wall = sum(usage.wall_s for usage in usages)
    cpu = sum(usage.cpu_s for usage in usages)
    gpu_series = pool_series(
        (usage.gpu_util for usage in usages),
        idle_threshold=1.0,
    )
    gpu_values = [usage.gpu_util_mean for usage in usages if usage.gpu_util_mean is not None]
    gpu_weighted = (
        sum(
            usage.gpu_util_mean * usage.wall_s
            for usage in usages
            if usage.gpu_util_mean is not None
        )
        / sum(usage.wall_s for usage in usages if usage.gpu_util_mean is not None)
        if gpu_values
        else None
    )
    return ResourceUsage(
        wall_s=wall,
        cpu_s=cpu,
        peak_rss_mb=max(usage.peak_rss_mb for usage in usages),
        rss_peak_mb=max(
            (usage.rss_peak_mb for usage in usages if usage.rss_peak_mb is not None),
            default=None,
        ),
        process_peak_rss_mb=max(
            (
                usage.process_peak_rss_mb
                for usage in usages
                if usage.process_peak_rss_mb is not None
            ),
            default=None,
        ),
        gpu_mb=max(
            (usage.gpu_mb for usage in usages if usage.gpu_mb is not None),
            default=None,
        ),
        gpu_util_mean=gpu_series.mean if gpu_series else gpu_weighted,
        gpu_util_peak=(
            gpu_series.max
            if gpu_series
            else max(
                (usage.gpu_util_peak for usage in usages if usage.gpu_util_peak is not None),
                default=None,
            )
        ),
        cpu=pool_series((usage.cpu for usage in usages), idle_threshold=0.05),
        rss=pool_series(usage.rss for usage in usages),
        gpu_util=gpu_series,
        gpu_mem=pool_series(usage.gpu_mem for usage in usages),
    )


def _subtitle_paths(out: Path, results: list[TranscriptionResult], srt: bool, vtt: bool) -> None:
    """Write per-file subtitles next to the JSONL, if asked and if timed."""
    if not (srt or vtt):
        return
    timed = [r for r in results if r.segments]
    if not timed:
        console.print(
            "[yellow]No subtitles written: this backend produced no timings. "
            "Re-run with --align to recover them.[/yellow]"
        )
        return
    for r in timed:
        stem = out.parent / f"{out.stem}-{Path(r.audio_path).stem}"
        if srt:
            write_srt(r, stem.with_suffix(".srt"))
        if vtt:
            write_vtt(r, stem.with_suffix(".vtt"))
    console.print(f"[dim]subtitles for {len(timed)} file(s) → {out.parent}/[/dim]")


def _record_alignment_provenance(result: TranscriptionResult, metadata: dict[str, Any]) -> None:
    """Attach alignment identity and make incomplete alignment fail closed."""
    result.metadata["alignment"] = metadata
    issues = list(result.trust_issues)
    issues.extend(str(issue) for issue in metadata.get("trust_issues", []))
    result.trust_issues = list(dict.fromkeys(issues))
    result.trusted = bool(
        result.trusted and metadata.get("trusted", False) and not result.trust_issues
    )


def _add_alignment(results: list[TranscriptionResult], device: str = "cpu") -> None:
    """Fill in timings for results whose backend could not supply any.

    Only touches results that need it, so a backend with native timestamps
    keeps its own — they are measured, whereas these are inferred.
    """
    from stt.align import align, alignment_metadata, load_aligner

    pending = [r for r in results if not r.error and r.text.strip() and not r.segments]
    if not pending:
        return

    with console.status(f"Aligning {len(pending)} transcript(s)…"):
        try:
            aligner = load_aligner(device)
        except Exception as exc:  # noqa: BLE001 - retain a diagnostic artifact
            error = f"{type(exc).__name__}: {exc}"
            metadata = alignment_metadata(None, device=device, status="unavailable", error=error)
            for r in pending:
                r.metadata["align_error"] = error
                _record_alignment_provenance(r, metadata)
            console.print(f"[yellow]aligner unavailable: {exc}[/yellow]")
            return
        for r in pending:
            try:
                segments = align(r.text, Path(r.audio_path), device=device, aligner=aligner)
            except Exception as exc:  # noqa: BLE001 - alignment is best-effort
                error = f"{type(exc).__name__}: {exc}"
                r.segments = None
                r.metadata["align_error"] = error
                metadata = alignment_metadata(aligner, device=device, status="failed", error=error)
                _record_alignment_provenance(r, metadata)
                console.print(f"[yellow]align failed for {Path(r.audio_path).name}: {exc}[/yellow]")
            else:
                r.segments = segments or None
                status = "completed" if segments else "empty"
                metadata = alignment_metadata(aligner, device=device, status=status)
                _record_alignment_provenance(r, metadata)


def _report_loops(results: list[TranscriptionResult]) -> None:
    """Flag decoder degeneration, which CER barely penalises but users notice.

    A sentence emitted four times costs a few percent of CER and destroys the
    transcript. With segments we can also say *when* it happened.
    """
    from stt.quality import find_loops, loop_summary

    for r in results:
        if r.error or not r.text.strip():
            continue
        sites = find_loops(r.text)
        if not sites:
            continue
        r.metadata["loops"] = [{"text": s.text, "count": s.count} for s in sites]
        where = ""
        if r.segments:
            hits = [s.start for s in r.segments if sites[0].text[:12] in s.text]
            if hits:
                where = f", first at {hits[0]:.0f}s"
        console.print(f"[yellow]⚠ {Path(r.audio_path).name}: {loop_summary(sites)}{where}[/yellow]")


def _slug(text: str) -> str:
    """Filesystem-safe model name for a default output path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "model"


def _prepare(paths: list[Path], limit: int, convert: bool) -> list[audio_mod.PreparedAudio]:
    files, skipped = audio_mod.find_audio(paths)
    if skipped:
        names = ", ".join(p.name for p in skipped[:3])
        more = f" (+{len(skipped) - 3} more)" if len(skipped) > 3 else ""
        console.print(
            f"[yellow]skipped {len(skipped)} unrecognised file(s): {names}{more}[/yellow]\n"
            "[dim]Pass a file directly to transcribe it regardless of extension.[/dim]"
        )
    if not files:
        raise typer.BadParameter("No audio files found")
    if limit:
        files = files[:limit]
    return [audio_mod.prepare_audio(f, DEFAULT_CACHE, convert=convert) for f in files]


def _mark_run_trust(results: list[TranscriptionResult], *, extra_issue: str | None = None) -> bool:
    """Compatibility wrapper for callers that imported the old CLI helper."""
    return mark_run_trust(results, extra_issue=extra_issue)


def _run_backend(
    backend_name: str,
    model: str | None,
    files: list[audio_mod.PreparedAudio],
    language: str | None,
    batch_size: int,
    options: dict,
) -> list[TranscriptionResult]:
    if batch_size < 1:
        raise typer.BadParameter("batch_size must be at least 1")
    cls = get_backend(backend_name)
    ok, reason = cls.is_available()
    if not ok:
        raise typer.BadParameter(
            f"Backend {backend_name!r} is not available: {reason}. Install with: {cls.install_hint}"
        )

    kwargs = {k: v for k, v in options.items() if v is not None}
    instance = cls(model, **kwargs) if model else cls(**kwargs)
    environment = bench_mod.capture_environment()
    provenance_issue: str | None = None
    binding = None
    try:
        binding = preflight_model_binding(
            backend_name,
            model,
            {**kwargs, "language": language, "batch_size": batch_size},
            environment=environment,
        )
        bind_model = getattr(instance, "bind_model", None)
        if not callable(bind_model):
            raise ProvenanceError("backend cannot accept an immutable model binding")
        bind_model(binding)
    except (KeyError, OSError, ProvenanceError, RuntimeError, TypeError, ValueError) as exc:
        provenance_issue = f"model provenance preflight failed: {exc}"
        binding = None
    instance._provenance_environment = environment

    # A bound model must be checked before a backend can open its weights. This
    # is especially important for GGUF, where ``load`` creates the native
    # session directly from the bound artifact path. Preflight or binding
    # attachment failures leave ``binding`` unset and retain the diagnostic
    # downloader path.
    if binding is not None:
        binding_issues = validate_binding(binding)
        if binding_issues:
            raise RuntimeError(
                "model binding validation failed before load: " + "; ".join(binding_issues)
            )

    console.print(
        f"[bold]{backend_name}[/bold] · model=[cyan]{instance.model}[/cyan] · "
        f"{len(files)} file(s) · lang={language or 'auto'}"
    )
    load_usage: ResourceUsage | None = None
    with console.status("Loading model…"):
        with measure() as measured_load:
            instance.load()
        load_usage = measured_load[0]
    if binding is not None:
        binding_issues = validate_binding(binding)
        if binding_issues:
            provenance_issue = "model binding changed during load: " + "; ".join(binding_issues)

    host = describe_host()
    try:
        with console.status("Transcribing corpus…"):
            results, _ = transcribe_corpus(
                instance,
                backend_name,
                files,
                language,
                batch_size,
                profile=True,
                load_usage=load_usage,
                host=host,
                require_provenance=True,
            )
    finally:
        instance.unload()
    if binding is not None:
        binding_issues = validate_binding(binding)
        if binding_issues:
            provenance_issue = "model binding changed during execution: " + "; ".join(
                binding_issues
            )
    mark_run_trust(
        results,
        extra_issue=provenance_issue,
        require_provenance=True,
    )
    return results


@app.command()
def transcribe(
    audio: Annotated[list[Path], typer.Argument(help="Audio files or directories")],
    backend: Annotated[str, typer.Option("--backend", "-b", help="Backend name")] = "omniasr-gguf",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name or card")] = None,
    language: Annotated[str, typer.Option("--language", "-l", help="Language code")] = BURMESE,
    limit: Annotated[int, typer.Option(help="Only the first N files; 0 for all")] = 0,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="JSONL path")] = None,
    device: Annotated[
        str | None, typer.Option(help="omniasr-torch only: auto, cpu, mps, cuda")
    ] = None,
    dtype: Annotated[
        str | None, typer.Option(help="omniasr-torch only: auto, float32, bfloat16")
    ] = None,
    threads: Annotated[
        int | None, typer.Option(help="omniasr-gguf only: ggml thread count")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Files per forward pass")] = 1,
    no_convert: Annotated[
        bool, typer.Option("--no-convert", help="Skip 16 kHz mono normalisation")
    ] = False,
    show: Annotated[bool, typer.Option("--show/--no-show", help="Print transcripts")] = True,
    do_align: Annotated[
        bool,
        typer.Option(
            "--align/--no-align",
            help="Recover timestamps by forced alignment when the backend has none",
        ),
    ] = False,
    srt: Annotated[bool, typer.Option("--srt", help="Also write SubRip subtitles")] = False,
    vtt: Annotated[bool, typer.Option("--vtt", help="Also write WebVTT subtitles")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show the runtime's own native logs")
    ] = False,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Write an explicitly untrusted diagnostic run instead of failing",
        ),
    ] = False,
) -> None:
    """Transcribe audio with one backend."""
    files = _prepare(audio, limit, convert=not no_convert)

    results = _run_backend(
        backend,
        model,
        files,
        language,
        batch_size,
        {"device": device, "dtype": dtype, "n_threads": threads, "verbose": verbose or None},
    )

    if allow_partial:
        for result in results:
            result.trusted = False
            if "partial run explicitly allowed" not in result.trust_issues:
                result.trust_issues.append("partial run explicitly allowed")

    if do_align or srt or vtt:
        _add_alignment(results)

    resolved_model = results[0].model if results else (model or backend)
    out = output or DEFAULT_OUTPUT_DIR / f"{backend}-{_slug(resolved_model)}.jsonl"
    write_jsonl(results, out)
    write_text(results, out.with_suffix(".txt"))
    _subtitle_paths(out, results, srt, vtt)
    _report_loops(results)

    if show:
        for r in results:
            name = Path(r.audio_path).name
            if r.error:
                console.print(f"[red]{name}[/red]: {r.error}")
            else:
                console.print(f"[dim]{name}[/dim]  [dim](RTF {_fmt(r.rtf, '.2f')})[/dim]")
                console.print(f"  {r.text}")

    rtf = _common_wall_rtf(results, allow_incomplete=allow_partial)
    failed = sum(1 for r in results if r.error)
    incomplete = failed or sum(not r.trusted for r in results)
    console.print(
        f"\n[green]{len(results) - failed}/{len(results)} transcribed[/green] · "
        f"{'attempted ' if incomplete else ''}corpus RTF {_fmt(rtf, '.2f')} → [bold]{out}[/bold]"
    )
    _report_resources(results)
    if incomplete:
        if allow_partial:
            console.print("[yellow]partial diagnostic artifact: not trusted evidence[/yellow]")
        else:
            raise typer.Exit(1)


def _normalized_path(path: str | Path) -> Path:
    """Normalize a path without requiring a legacy result's file to exist."""
    return Path(path).expanduser().resolve(strict=False)


def _same_audio(recorded: str, source: Path, prepared: Path) -> bool:
    """Whether a legacy result names this exact source or prepared file."""
    if not recorded:
        return False
    recorded_path = _normalized_path(recorded)
    return recorded_path in {_normalized_path(source), _normalized_path(prepared)}


def _refuse_to_clobber_another_run(path: Path, result: TranscriptionResult) -> None:
    """Refuse to overwrite a JSONL that belongs to a different run.

    `stt align -o` writes a record whose backend is "align" and whose model is
    the aligner. Pointed at a name a transcription run already owns, it
    replaces that run's identity in place: the file keeps the transcript but
    loses the backend, model and timings that said where it came from. That is
    exactly how outputs/eternity-7b.jsonl stopped being a 7B run, and nothing
    reported it until the manifest was audited months later.
    """
    if not path.is_file():
        return
    try:
        existing = read_jsonl(path)
    except (OSError, ValueError):
        return  # Unreadable: not something to protect.
    conflicting = sorted(
        {
            (record.backend, record.model)
            for record in existing
            if (record.backend, record.model) != (result.backend, result.model)
        }
    )
    if not conflicting:
        return
    owners = ", ".join(f"{backend}/{model}" for backend, model in conflicting)
    raise typer.BadParameter(
        f"{path} already holds a run from {owners}, and this command would "
        f"replace it with {result.backend}/{result.model}. Pass --output with a "
        "different name; the transcription run's provenance is not recoverable "
        "once overwritten."
    )


@app.command("align")
def align_cmd(
    audio: Annotated[Path, typer.Argument(help="Audio file to align against")],
    text: Annotated[
        Path | None, typer.Option("--text", "-t", help="Transcript file (UTF-8)")
    ] = None,
    results: Annotated[
        Path | None, typer.Option("--results", "-r", help="JSONL from a previous run")
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output stem")] = None,
    device: Annotated[str, typer.Option(help="cpu, mps or cuda")] = "cpu",
    srt: Annotated[bool, typer.Option("--srt/--no-srt", help="Write SubRip")] = True,
    vtt: Annotated[bool, typer.Option("--vtt", help="Also write WebVTT")] = False,
) -> None:
    """Time an existing transcript against its audio.

    Gives timestamps and per-segment confidence to any transcript, including
    ones from backends that cannot produce them, which is what makes subtitles
    and per-region error reporting possible.
    """
    from stt.align import align as align_text
    from stt.align import alignment_metadata, load_aligner

    if (text is None) == (results is None):
        raise typer.BadParameter("Pass exactly one of --text or --results")

    prepared = audio_mod.prepare_audio(audio, DEFAULT_CACHE)
    source_result: TranscriptionResult | None = None
    if text is not None:
        transcript = text.read_text(encoding="utf-8")
    else:
        assert results is not None
        loaded = read_jsonl(results)
        matching = [r for r in loaded if r.audio_id == prepared.audio_id]
        if len(matching) > 1:
            raise typer.BadParameter(
                f"Ambiguous results in {results}: {len(matching)} records share verified "
                f"audio_id for {audio}. Remove duplicates before aligning."
            )
        if not matching:
            # Alignment remains useful for inspecting legacy output, but a
            # path-based fallback cannot create trusted provenance.
            matching = [
                r
                for r in loaded
                if not r.audio_id
                and _same_audio(r.audio_path, prepared.source_path, prepared.prepared_path)
            ]
            if len(matching) > 1:
                raise typer.BadParameter(
                    f"Ambiguous legacy results in {results}: {len(matching)} exact path matches "
                    f"for {audio}. Use a result with a verified audio_id."
                )
        if not matching:
            known = ", ".join(sorted({Path(r.audio_path).stem for r in loaded})[:3])
            raise typer.BadParameter(f"No result in {results} for {audio.stem!r}. Found: {known}")
        source_result = matching[0]
        transcript = source_result.text

    with console.status("Aligning…"):
        try:
            aligner = load_aligner(device)
            segments = align_text(
                transcript,
                prepared.prepared_path,
                device=device,
                aligner=aligner,
            )
        except Exception as exc:  # noqa: BLE001 - keep the command diagnostic-friendly
            console.print(f"[red]Alignment failed: {exc}[/red]")
            raise typer.Exit(1) from exc

    if not segments:
        console.print("[red]Alignment produced no segments.[/red]")
        raise typer.Exit(1)

    alignment = alignment_metadata(aligner, device=device, status="completed")
    if source_result is not None:
        metadata = {**source_result.metadata, "alignment": alignment}
        issues = list(source_result.trust_issues)
        if source_result.audio_id != prepared.audio_id:
            issues.append("alignment source was matched by legacy path without verified audio_id")
        issues.extend(str(issue) for issue in alignment.get("trust_issues", []))
        result = replace(
            source_result,
            segments=segments,
            metadata=metadata,
            trusted=bool(source_result.trusted and alignment.get("trusted", False) and not issues),
            trust_issues=list(dict.fromkeys(issues)),
        )
    else:
        model_provenance = None
        # Keep diagnostic callers that still return the historical tuple API
        # usable; alignment_metadata marks that path untrusted.
        result_provenance = getattr(aligner, "result_provenance", None)
        if result_provenance is not None:
            try:
                model_provenance = result_provenance.to_dict()
            except Exception as exc:  # noqa: BLE001 - metadata remains explicitly untrusted
                alignment["trusted"] = False
                alignment["trust_issues"] = list(
                    dict.fromkeys(
                        [
                            *alignment.get("trust_issues", []),
                            f"aligner provenance serialization failed: {type(exc).__name__}: {exc}",
                        ]
                    )
                )
        alignment_trust_issues = list(alignment.get("trust_issues", []))
        result = TranscriptionResult(
            audio_path=str(prepared.prepared_path),
            source_path=str(prepared.source_path),
            source_sha256=prepared.source_sha256,
            reference_id=prepared.reference_id,
            audio_id=prepared.audio_id,
            text=transcript,
            backend="align",
            model="mms-1b-all",
            audio_duration_s=audio_mod.duration_of(prepared.prepared_path),
            segments=segments,
            metadata={"alignment": alignment},
            trusted=bool(alignment.get("trusted", False) and not alignment_trust_issues),
            trust_issues=alignment_trust_issues,
            model_provenance=model_provenance,
        )
    stem = output or DEFAULT_OUTPUT_DIR / f"{audio.stem}"
    _refuse_to_clobber_another_run(stem.with_suffix(".jsonl"), result)
    if srt:
        console.print(f"{write_srt(result, stem.with_suffix('.srt'))} cues → {stem}.srt")
    if vtt:
        console.print(f"{write_vtt(result, stem.with_suffix('.vtt'))} cues → {stem}.vtt")
    write_jsonl([result], stem.with_suffix(".jsonl"))

    confidences = [s.confidence for s in segments if s.confidence is not None]
    mean_conf = sum(confidences) / len(confidences) if confidences else None
    weak = sum(1 for c in confidences if c < 0.5)
    console.print(
        f"[green]{len(segments)} segments[/green] · "
        f"mean confidence {_fmt(mean_conf, '.3f')} · {weak} below 0.5"
    )


@app.command("eval")
def eval_cmd(
    results_path: Annotated[Path, typer.Argument(help="JSONL produced by `stt transcribe`")],
    reference: Annotated[Path, typer.Option("--reference", "-r", help="Two-column TSV")],
    keep_whitespace: Annotated[
        bool, typer.Option("--keep-whitespace", help="Do not strip spaces before scoring")
    ] = False,
    keep_punctuation: Annotated[
        bool, typer.Option("--keep-punctuation", help="Do not strip punctuation")
    ] = False,
    per_file: Annotated[bool, typer.Option("--per-file", help="Show a row per clip")] = False,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Print explicitly partial diagnostic metrics and exit successfully",
        ),
    ] = False,
) -> None:
    """Score a transcription run against reference transcripts (CER)."""
    results = read_jsonl(results_path)
    refs = load_references(reference)
    opts = NormalizeOptions(
        strip_whitespace=not keep_whitespace,
        strip_punctuation=not keep_punctuation,
    )
    score = score_results(results, refs, opts)

    if per_file:
        table = Table(title=f"{score.backend} · {score.model}")
        table.add_column("File")
        table.add_column("CER", justify="right")
        table.add_column("Chars", justify="right")
        table.add_column("Note", style="red")
        for item in score.items:
            table.add_row(
                Path(item.audio_path).name,
                _fmt(item.cer),
                str(item.ref_chars),
                item.error or "",
            )
        console.print(table)

    partial = not score.complete
    if partial and not allow_partial:
        cer = wer = rtf = None
    else:
        cer = score.partial_cer if partial else score.cer
        wer = score.partial_wer if partial else score.wer
        rtf = score.partial_rtf if partial else score.rtf

    label = "partial diagnostic" if partial else "trusted"
    style = "yellow" if partial else "green"
    console.print(
        f"\n[bold]{score.backend}[/bold] · {score.model} · [{style}]{label}[/{style}]\n"
        f"  Coverage   {score.n_scored}/{score.total} ({score.coverage:.1%})\n"
        f"  CER   [bold cyan]{_fmt(cer)}[/bold cyan]\n"
        f"  WER   {_fmt(wer)}   [dim](not meaningful for Burmese)[/dim]\n"
        f"  RTF   {_fmt(rtf, '.2f')}"
    )
    if score.trust_issues:
        console.print("[yellow]  Issues: " + "; ".join(score.trust_issues) + "[/yellow]")
    if partial and not allow_partial:
        console.print(
            "[red]Trusted metrics suppressed. Re-run with --allow-partial for diagnostics.[/red]"
        )
        raise typer.Exit(1)


@app.command()
def evidence(
    check: Annotated[
        bool,
        typer.Option("--check", help="Check manifest-owned Markdown blocks (the default)"),
    ] = False,
    update: Annotated[
        bool,
        typer.Option("--update", help="Rewrite the manifest-owned Markdown blocks"),
    ] = False,
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Evidence manifest JSON path"),
    ] = Path("evidence/manifest.json"),
) -> None:
    """Check or update artifact-backed measured documentation."""
    if check and update:
        raise typer.BadParameter("--check and --update cannot be combined")
    try:
        result = update_evidence(manifest) if update else check_evidence(manifest)
    except EvidenceError as exc:
        console.print(f"[red]Evidence validation failed: {exc}[/red]")
        raise typer.Exit(1) from None

    if result.issues:
        for issue in result.issues:
            console.print(f"[yellow]{issue}[/yellow]")
    if result.publishable:
        console.print(f"[green]publishable evidence[/green] → {result.document}")
    else:
        console.print(
            f"[yellow]documentation synchronized but unverified[/yellow] → {result.document}"
        )
    if not result.matches:
        console.print(
            "[red]Documentation differs from generated evidence. Run `stt evidence --update`.[/red]"
        )
        raise typer.Exit(1)


@app.command()
def vote(
    runs: Annotated[
        list[Path],
        typer.Argument(help="JSONL runs to combine. Put your most accurate model FIRST."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the result")],
    weight: Annotated[
        list[str] | None,
        typer.Option("--weight", "-w", help="model=value, repeatable. Default: measured ranking"),
    ] = None,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Omit unavailable non-pivot voters and write untrusted diagnostics",
        ),
    ] = False,
) -> None:
    """Combine several transcription runs by weighted per-character vote.

    Different systems fail on different words, so voting recovers accuracy no
    single model reaches (docs/findings.md#voting). The first run given is the
    pivot and should be your best model — voting can only correct characters
    the pivot proposed.
    """
    if len(runs) < 2:
        raise typer.BadParameter("need at least two runs to vote between")

    weights = dict(DEFAULT_WEIGHTS)
    for item in weight or []:
        name, _, value = item.partition("=")
        if not value:
            raise typer.BadParameter(f"--weight expects model=value, got {item!r}")
        weights[name.strip()] = float(value)

    # Keep the run order so the first remains the pivot. Identity validation
    # and grouping happen centrally before any output is written.
    loaded_runs: dict[str, list[TranscriptionResult]] = {}
    labels: list[str] = []
    for path in runs:
        results = read_jsonl(path)
        label = results[0].model if results else path.stem
        while label in labels:  # two runs of the same model: keep both distinct
            label += "'"
        labels.append(label)
        loaded_runs[label] = results

    pivot = labels[0]
    try:
        groups = prepare_vote_groups(loaded_runs, pivot, allow_partial=allow_partial)
    except VoteInputError as exc:
        raise typer.BadParameter(str(exc)) from exc

    combined: list[TranscriptionResult] = []
    for group in groups:
        base = group.pivot
        issues = [*group.trust_issues, _DERIVED_VOTE_TRUST_ISSUE]
        if allow_partial:
            issues.append("partial vote explicitly allowed")
        if base.error:
            combined.append(
                TranscriptionResult(
                    audio_path=base.audio_path,
                    source_path=base.source_path,
                    source_sha256=base.source_sha256,
                    reference_id=base.reference_id,
                    audio_id=group.audio_id,
                    text="",
                    backend="vote",
                    model="+".join(labels),
                    language=base.language,
                    elapsed_s=group.elapsed_s,
                    audio_duration_s=base.audio_duration_s,
                    error=f"pivot {pivot!r} failed: {base.error}",
                    metadata=group.provenance(),
                    trusted=False,
                    trust_issues=list(dict.fromkeys(issues)),
                    model_provenance=None,
                )
            )
            continue

        usable = group.usable_results
        texts = {name: result.text for name, result in usable.items()}
        combined.append(
            TranscriptionResult(
                audio_path=base.audio_path,
                source_path=base.source_path,
                source_sha256=base.source_sha256,
                reference_id=base.reference_id,
                audio_id=group.audio_id,
                text=rover(texts, pivot, weights),
                backend="vote",
                model="+".join(labels),
                language=base.language,
                elapsed_s=group.elapsed_s,
                audio_duration_s=base.audio_duration_s,
                metadata=group.provenance(),
                trusted=False,
                trust_issues=list(dict.fromkeys(issues)),
                model_provenance=None,
            )
        )

    write_jsonl(combined, output)
    console.print(
        f"voted {len(combined)} file(s) across {len(labels)} run(s) "
        f"[dim](pivot: {pivot})[/dim] → {output}"
    )
    if allow_partial:
        console.print("[yellow]partial diagnostic artifact: not trusted evidence[/yellow]")


def _confirm_download(cls) -> bool:
    """Ask before a backend's default model pulls a large checkpoint.

    ``stt compare`` runs every installed backend at its default model, and the
    omniasr-torch default is a 31 GB download. Starting that unannounced is a
    nasty surprise, so anything over a gigabyte that is not known to be cached
    gets a prompt.
    """
    try:
        probe = cls()
    except Exception as exc:  # noqa: BLE001 - construction may need arguments we lack
        # Cannot size the download, so ask rather than assume it is small: this
        # guard exists precisely to stop an unannounced multi-gigabyte fetch.
        return typer.confirm(
            f"{cls.name}: cannot check the download size ({exc}). Continue?", default=False
        )

    if probe.weights_cached() is True:
        return True
    size_mb = probe.estimated_download_mb()
    if size_mb is None or size_mb < 1024:
        return True

    return typer.confirm(
        f"{cls.name}: {probe.model} may need to download ~{size_mb / 1024:.1f} GB. Continue?",
        default=False,
    )


@app.command()
def compare(
    audio: Annotated[list[Path], typer.Argument(help="Audio files or directories")],
    reference: Annotated[
        Path | None, typer.Option("--reference", "-r", help="Two-column TSV for CER")
    ] = None,
    language: Annotated[str, typer.Option("--language", "-l")] = BURMESE,
    limit: Annotated[int, typer.Option(help="Only the first N files; 0 for all")] = 0,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Restrict to these backends")
    ] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_OUTPUT_DIR,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not prompt before large downloads")
    ] = False,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Files per backend batch")] = 1,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Keep available backend diagnostics when the comparison is incomplete",
        ),
    ] = False,
) -> None:
    """Run every installed backend over the same audio and tabulate the results."""
    if batch_size < 1:
        raise typer.BadParameter("batch_size must be at least 1")
    files = _prepare(audio, limit, convert=True)
    refs = load_references(reference) if reference else None

    known = set(all_backends())
    unknown = sorted(set(only or []) - known)
    if unknown:
        raise typer.BadParameter(f"unknown backend(s): {', '.join(unknown)}")

    table = Table(title="Backend comparison")
    table.add_column("Backend", style="bold")
    table.add_column("Model")
    table.add_column("CER", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Failed", justify="right")

    collected: list[tuple[str, list[TranscriptionResult]]] = []
    problems: list[str] = []
    for name, cls in all_backends().items():
        if only and name not in only:
            continue
        ok, reason = cls.is_available()
        if not ok:
            console.print(f"[yellow]skipping {name}[/yellow] — {reason}")
            problems.append(f"{name} unavailable: {reason}")
            continue

        if not yes and not _confirm_download(cls):
            console.print(f"[yellow]skipping {name}[/yellow] — download declined")
            problems.append(f"{name} download declined")
            continue

        try:
            results = _run_backend(name, None, files, language, batch_size, {})
        except Exception as exc:  # noqa: BLE001 - retain other diagnostic runs
            problems.append(f"{name} failed: {type(exc).__name__}: {exc}")
            console.print(f"[red]{problems[-1]}[/red]")
            continue
        failed = sum(1 for result in results if result.error)
        if failed:
            problems.append(f"{name}: {failed}/{len(results)} result(s) failed")
        collected.append((name, results))

    if not collected:
        console.print("[red]No backends available.[/red] Run `stt backends` to see why.")
        raise typer.Exit(1)

    scores = {name: score_results(results, refs) for name, results in collected if refs is not None}
    for name, score in scores.items():
        if not score.complete:
            problems.append(
                f"{name}: incomplete evaluation coverage ({score.n_scored}/{score.total} scored)"
            )

    if refs is None:
        for name, results in collected:
            untrusted = [result for result in results if not result.trusted or result.trust_issues]
            if not untrusted:
                continue
            details = sorted(
                {issue for result in untrusted for issue in result.trust_issues if issue}
            )
            detail = f"{name}: {len(untrusted)}/{len(results)} result(s) untrusted"
            if details:
                detail += f" ({'; '.join(details[:3])})"
            problems.append(detail)

    comparison_complete = not problems
    for name, results in collected:
        if not comparison_complete or allow_partial:
            issue = (
                "partial comparison explicitly allowed"
                if allow_partial
                else "comparison omitted or failed one or more expected backends"
            )
            for result in results:
                result.trusted = False
                if issue not in result.trust_issues:
                    result.trust_issues.append(issue)
        # Keyed on backend *and* model, for the same reason `stt transcribe`
        # is: keying on the backend alone silently overwrote one run with
        # another, and this command is the one the docs print under Reproduce.
        model_slug = _slug(results[0].model) if results else name
        write_jsonl(results, output_dir / f"{name}-{model_slug}.jsonl")

        score = scores.get(name)
        cer_value = None
        if score is not None:
            cer_value = (
                score.partial_cer if allow_partial else score.cer if comparison_complete else None
            )
        failed = sum(1 for result in results if result.error)
        model = results[0].model if results else "?"
        rtf = (
            _common_wall_rtf(results, allow_incomplete=True)
            if allow_partial
            else _common_wall_rtf(results)
            if comparison_complete
            else None
        )
        table.add_row(name, model, _fmt(cer_value), _fmt(rtf, ".2f"), str(failed))

    console.print()
    console.print(table)
    if not refs:
        console.print("[dim]Pass --reference to get CER instead of just timings.[/dim]")
    if problems:
        console.print("[yellow]" + "; ".join(problems) + "[/yellow]")
        if not allow_partial:
            raise typer.Exit(1)
    if allow_partial:
        console.print("[yellow]partial diagnostic artifacts: not trusted evidence[/yellow]")


@app.command()
def route(
    base: Annotated[Path, typer.Argument(help="Cheap model's run. Must carry timed segments.")],
    strong: Annotated[Path, typer.Argument(help="Expensive model's run over the same audio")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the result")],
    escalate: Annotated[
        float, typer.Option("--escalate", help="Share of audio duration to re-transcribe")
    ] = DEFAULT_ESCALATE,
    block: Annotated[
        int, typer.Option("--block", help="Segments per routing block; coarse on purpose")
    ] = DEFAULT_BLOCK,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Route verified covered items and write untrusted diagnostics",
        ),
    ] = False,
) -> None:
    """Take the strong model's text only where the cheap one was least confident.

    Both runs must already exist — this splices transcripts, it does not
    transcribe. Routing is by block *within* each file; the measured routing
    results were per whole clip, so the two are not directly comparable.
    See docs/findings.md#routing and #seam-tax.
    """
    from stt.audio import join_segments
    from stt.cascade import (
        RouteInputError,
        corpus_escalated_share,
        escalated_share,
        plan,
        prepare_route_pairs,
        stitch,
    )

    base_results = read_jsonl(base)
    strong_results = read_jsonl(strong)
    try:
        prepared = prepare_route_pairs(
            base_results,
            strong_results,
            allow_partial=allow_partial,
        )
    except RouteInputError as exc:
        console.print(f"[red]Cannot route: {exc}[/red]")
        raise typer.Exit(1) from None

    routed: list[TranscriptionResult] = []
    routed_intervals: list[tuple[float, list[Any]]] = []
    for pair in prepared.pairs:
        r, counterpart = pair.base, pair.strong
        assert r.segments is not None
        assert counterpart.segments is not None
        assert r.audio_duration_s is not None

        intervals = plan(
            r.segments,
            r.audio_duration_s,
            escalate=escalate,
            block_size=block,
        )
        merged = stitch(r.segments, counterpart.segments, intervals)
        share = escalated_share(r.audio_duration_s, intervals)
        routed_intervals.append((r.audio_duration_s, intervals))
        routed.append(
            TranscriptionResult(
                audio_path=r.audio_path,
                source_path=r.source_path,
                source_sha256=r.source_sha256,
                reference_id=r.reference_id,
                audio_id=pair.audio_id,
                text=join_segments(merged),
                backend="route",
                model=f"{r.model}+{counterpart.model}",
                language=r.language,
                audio_duration_s=r.audio_duration_s,
                segments=merged,
                trusted=False,
                trust_issues=list(dict.fromkeys((*pair.trust_issues, _DERIVED_ROUTE_TRUST_ISSUE))),
                model_provenance=None,
                metadata={
                    "base": r.model,
                    "strong": counterpart.model,
                    "escalated_share": round(share, 4),
                    "seams": len(intervals),
                },
            )
        )

    if prepared.skipped_audio_ids:
        console.print(
            f"[yellow]skipped {len(prepared.skipped_audio_ids)} unrouteable audio ID(s)[/yellow]"
        )
    if not routed:
        console.print("[red]Nothing to route.[/red]")
        raise typer.Exit(1)

    write_jsonl(routed, output)
    write_text(routed, output.with_suffix(".txt"))
    weighted_share = corpus_escalated_share(routed_intervals)
    seams = sum(r.metadata["seams"] for r in routed)
    console.print(
        f"[green]{len(routed)} file(s) routed[/green] · "
        f"{weighted_share:.0%} of audio escalated · {seams} seam(s) → [bold]{output}[/bold]"
    )
    if allow_partial:
        console.print("[yellow]partial diagnostic artifact: not trusted evidence[/yellow]")


def _hf_cache_entries() -> set[str]:
    """Hub directory names this project owns, for a selective migration.

    Models *and* datasets: `stt fetch-fleurs` caches the FLEURS archive under
    ``datasets--google--fleurs``, which is 896 MB and re-downloads if missed.
    """
    from stt.align import MMS_REPO
    from stt.backends.transformers_asr import MODELS as HF_MODELS
    from stt.datasets import FLEURS_REPO

    models = {spec.repo for spec in HF_MODELS.values()} | {MMS_REPO}
    names = {f"models--{repo.replace('/', '--')}" for repo in models}
    return names | {f"datasets--{FLEURS_REPO.replace('/', '--')}"}


def _legacy_stores() -> list[tuple[str, Path, Path]]:
    """``(label, old, new)`` for caches that may still sit under ``~/.cache``."""
    from stt.paths import cache_root

    home, root = Path.home() / ".cache", cache_root()
    return [
        (name, home / name, root / name)
        for name in ("dolphin", "crispasr", "stt")
        if (home / name).is_dir() and (home / name).resolve() != (root / name).resolve()
    ]


def _stray_hf_entries() -> list[tuple[Path, Path]]:
    """This project's HF snapshots still in the shared per-user cache.

    Only entries this project names are listed. The shared cache normally holds
    other projects' models too, and moving those would break them.
    """
    from stt.paths import cache_root

    old_hub = Path.home() / ".cache" / "huggingface" / "hub"
    new_hub = cache_root() / "huggingface" / "hub"
    if not old_hub.is_dir() or old_hub.resolve() == new_hub.resolve():
        return []
    wanted = _hf_cache_entries()
    return [(d, new_hub / d.name) for d in sorted(old_hub.iterdir()) if d.name in wanted]


@app.command()
def doctor(
    migrate: Annotated[
        bool, typer.Option("--migrate", help="Move this project's caches into the cache root")
    ] = False,
) -> None:
    """Check the environment: runtimes, prerequisites, and where weights live."""
    import shutil

    from stt.paths import ENV_VAR, FAIRSEQ2_HOME, RUNTIME_DIRS, cache_root, directory_size_mb
    from stt.paths import fairseq2_status as fs_status

    root = cache_root()
    console.print(f"checkout   [bold]{Path.cwd()}[/bold]")
    console.print(
        f"cache root [bold]{root}[/bold]"
        + (f"  [dim](from ${ENV_VAR})[/dim]" if os.environ.get(ENV_VAR) else "")
    )

    tools = Table(title="Prerequisites")
    tools.add_column("", style="bold")
    tools.add_column("")
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        tools.add_row(tool, f"[green]{found}[/green]" if found else "[red]missing[/red]")
    tools.add_row("python", sys.version.split()[0])
    console.print(tools)

    ready = 0
    backend_table = Table(title="Backends")
    backend_table.add_column("Name", style="bold")
    backend_table.add_column("Status")
    for name, cls in all_backends().items():
        ok, reason = cls.is_available()
        ready += ok
        backend_table.add_row(
            name, f"[green]ready[/green] {reason}" if ok else f"[red]missing[/red] {reason}"
        )
    console.print(backend_table)
    if ready < len(all_backends()):
        console.print("[yellow]Install everything with: ./scripts/bootstrap.sh[/yellow]")

    weights = Table(title="Weight caches")
    weights.add_column("Runtime", style="bold")
    weights.add_column("Location")
    weights.add_column("Size", justify="right")
    for name in RUNTIME_DIRS:
        path = root / name
        size = directory_size_mb(path)
        weights.add_row(
            name,
            str(path) if size is not None else "[dim]not created[/dim]",
            f"{size / 1024:.1f} GB"
            if size and size >= 1024
            else (f"{size:.0f} MB" if size else "—"),
        )
    console.print(weights)

    state, target = fs_status()
    if state == "linked":
        console.print(f"[green]fairseq2 → {target}[/green] [dim](repo-local via symlink)[/dim]")
    elif state == "local":
        size = directory_size_mb(FAIRSEQ2_HOME) or 0
        console.print(
            f"[yellow]fairseq2 still at {FAIRSEQ2_HOME}[/yellow] "
            f"[dim]({size / 1024:.1f} GB — it reads that path directly and has no "
            f"env var; --migrate can relink it)[/dim]"
        )
    elif state == "elsewhere":
        console.print(f"[yellow]fairseq2 → {target}[/yellow] [dim](not this cache root)[/dim]")

    strays = _legacy_stores()
    hf_strays = _stray_hf_entries()
    if not migrate:
        pending = len(strays) + len(hf_strays) + (state == "local")
        if pending:
            console.print(
                f"\n[yellow]{pending} cache(s) still outside the cache root.[/yellow] "
                "Run [bold]stt doctor --migrate[/bold] to move them."
            )
        else:
            console.print("\n[green]All caches are in the cache root.[/green]")
        return

    moved = 0
    for label, old, new in strays:
        console.print(f"moving {label}: {old} → {new}")
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        moved += 1

    if hf_strays:
        console.print(
            f"[dim]{len(hf_strays)} Hugging Face model(s) belong to this project; "
            "anything else in the shared cache is left alone.[/dim]"
        )
    for old, new in hf_strays:
        console.print(f"moving {old.name}")
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        moved += 1

    if state == "local":
        size = directory_size_mb(FAIRSEQ2_HOME) or 0
        if typer.confirm(
            f"Move {size / 1024:.1f} GB of fairseq2 checkpoints into {root / 'fairseq2'} "
            f"and replace {FAIRSEQ2_HOME} with a symlink?",
            default=False,
        ):
            destination = root / "fairseq2"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with console.status(f"Moving {size / 1024:.1f} GB…"):
                shutil.move(str(FAIRSEQ2_HOME), str(destination))
                FAIRSEQ2_HOME.symlink_to(destination)
            console.print(f"[green]fairseq2 → {destination}[/green]")
            moved += 1
        else:
            console.print("[dim]fairseq2 left where it is.[/dim]")

    console.print(f"\n[green]{moved} cache(s) migrated.[/green]" if moved else "\nNothing to move.")


@app.command()
def bench(
    update: Annotated[
        bool, typer.Option("--update", help="Rewrite the baseline instead of checking it")
    ] = False,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Restrict to these backends")
    ] = None,
    baseline_path: Annotated[
        Path, typer.Option("--baseline", help="Baseline file")
    ] = bench_mod.BASELINE,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Files per backend batch")] = 1,
    warmups: Annotated[int, typer.Option(help="Untimed complete-corpus warmups per worker")] = 1,
    repeats: Annotated[int, typer.Option(help="Measured complete-corpus repeats per worker")] = 1,
    workers: Annotated[int, typer.Option(help="Fresh worker processes per subject")] = 1,
    profile: Annotated[
        bool,
        typer.Option("--profile/--no-profile", help="Collect timestamped resource samples"),
    ] = False,
    artifact_dir: Annotated[
        Path, typer.Option("--artifacts", help="Raw measurement artifact directory")
    ] = bench_mod.ARTIFACT_DIR,
    timeout_s: Annotated[
        float, typer.Option("--timeout", help="Maximum seconds for one subject worker")
    ] = 7_200,
    seed: Annotated[int, typer.Option("--seed", help="Counterbalanced session schedule seed")] = 0,
    verify: Annotated[
        bool, typer.Option("--verify", help="Verify a stored baseline without loading models")
    ] = False,
    accept_transcript_changes: Annotated[
        bool,
        typer.Option(
            "--accept-transcript-changes",
            help="Allow an intentional transcript change during baseline update",
        ),
    ] = False,
    approval_note: Annotated[
        str | None, typer.Option("--approval-note", help="Required rationale for text changes")
    ] = None,
) -> None:
    """Measure the fixed clip set and compare against the committed baseline.

    A default run is a transcript-only smoke diagnostic. Trusted timing gates
    require the complete v2 session protocol and immutable provenance.
    """
    if verify:
        if update or only:
            raise typer.BadParameter("--verify cannot be combined with --update or --only")
        try:
            issues = bench_mod.verify_baseline(baseline_path)
        except Exception as exc:  # noqa: BLE001 - surface corrupt artifacts as a failed check
            console.print(f"[red]baseline verification failed: {exc}[/red]")
            raise typer.Exit(1) from exc
        if issues:
            for issue in issues:
                console.print(f"[red]{issue}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]baseline verified[/green] → {baseline_path}")
        return
    if batch_size < 1:
        raise typer.BadParameter("batch_size must be at least 1")
    if update and only:
        raise typer.BadParameter("--update cannot be combined with --only")
    if accept_transcript_changes and not update:
        raise typer.BadParameter("--accept-transcript-changes requires --update")
    if accept_transcript_changes and (not approval_note or not approval_note.strip()):
        raise typer.BadParameter("--approval-note is required when accepting transcript changes")
    if warmups < 0 or repeats < 1 or workers < 1 or timeout_s <= 0:
        raise typer.BadParameter(
            "warmups must be non-negative; repeats, workers, and timeout must be positive"
        )
    if update and (warmups < 3 or repeats < 3 or workers < 5 or profile):
        raise typer.BadParameter(
            "baseline update requires --warmups 3 --repeats 3 --workers 5 --no-profile"
        )
    if update and bench_mod.capture_environment().get("git_dirty"):
        raise typer.BadParameter(
            "baseline update requires a clean source worktree; use a diagnostic run first"
        )
    known_backends = {backend for backend, _ in bench_mod.SUBJECTS}
    unknown = sorted(set(only or []) - known_backends)
    if unknown:
        raise typer.BadParameter(f"unknown benchmark backend(s): {', '.join(unknown)}")
    existing_baseline = bench_mod.load_baseline(baseline_path)
    if existing_baseline and existing_baseline.get("artifact_kind") == "baseline-v2":
        verification_issues = bench_mod.verify_baseline(baseline_path)
        if verification_issues:
            detail = "; ".join(verification_issues)
            raise typer.BadParameter(f"stored baseline failed verification: {detail}")
    files = bench_mod.clips()
    prepared = [audio_mod.prepare_audio(f, DEFAULT_CACHE) for f in files]
    inputs = [AudioInput.from_prepared(item) for item in prepared]
    reference_ids = [item.reference_id for item in inputs]
    run_id = new_run_id()
    run_dir = artifact_dir / run_id

    selected_subjects = [
        (backend_name, model_name)
        for backend_name, model_name in bench_mod.SUBJECTS
        if not only or backend_name in only
    ]
    labels = [
        f"{backend_name}/{model_name}" if model_name else backend_name
        for backend_name, model_name in selected_subjects
    ]
    schedule = bench_mod.counterbalanced_schedule(labels, workers, seed)
    measurement: dict[str, object] = {
        "artifact_kind": "measurement-v3",
        "schema_version": bench_mod.MEASUREMENT_SCHEMA_VERSION,
        "experiment_id": run_id,
        "run_id": run_id,
        "host": bench_mod.host_signature(),
        "clips": reference_ids,
        "corpus": [
            {
                "audio_id": item.audio_id,
                "reference_id": item.reference_id,
                "duration_s": item.duration_s,
            }
            for item in inputs
        ],
        "batch_size": batch_size,
        "schedule_seed": seed,
        "session_schedule": schedule,
        "protocol": {
            "warmups": warmups,
            "repeats": repeats,
            "workers": workers,
            "profile": profile,
        },
        "raw_artifact_dir": str(run_dir),
        "runs": [],
    }

    subject_info: dict[str, dict[str, object]] = {}
    disabled: set[str] = set()
    preflight_environment = bench_mod.capture_environment()
    execution_options = {"language": BURMESE, "batch_size": batch_size}
    for backend_name, model_name in selected_subjects:
        subject = f"{backend_name}/{model_name}" if model_name else backend_name
        cls = get_backend(backend_name)
        ok, reason = cls.is_available()
        subject_info[subject] = {
            "backend": backend_name,
            "model": model_name,
            "available": ok,
            "reason": reason,
            "responses": [],
            "raw_artifacts": [],
            "binding": None,
            "preflight_issues": [],
        }
        if not ok:
            disabled.add(subject)
            console.print(f"[red]{subject} unavailable[/red] — {reason}")
            continue
        try:
            subject_info[subject]["binding"] = preflight_model_binding(
                backend_name,
                model_name,
                execution_options,
                environment=preflight_environment,
            )
        except (ProvenanceError, OSError, RuntimeError, ValueError) as exc:
            issue = f"model preflight unavailable: {exc}"
            subject_info[subject]["preflight_issues"] = [issue]
            console.print(f"[yellow]{subject} provenance unavailable[/yellow] — {exc}")

    for session_index, session in enumerate(schedule):
        for subject, launch_position in session:
            info = subject_info[subject]
            if subject in disabled:
                continue
            request = WorkerRequest(
                run_id=run_id,
                subject=SubjectSpec(
                    str(info["backend"]),
                    info["model"],
                    BURMESE,
                    batch_size,
                    options=dict(execution_options),
                ),
                inputs=tuple(inputs),
                warmups=warmups,
                repeats=repeats,
                profile=profile,
                worker_index=session_index,
                experiment_id=run_id,
                session_id=f"{run_id}:session-{session_index}",
                session_index=session_index,
                launch_position=launch_position,
                schedule_seed=seed,
                model_binding=info["binding"],
            )
            console.print(
                f"[bold]{subject}[/bold] · session {session_index + 1}/{workers} "
                f"· position {launch_position + 1}/{len(selected_subjects)}"
            )
            response, response_path = bench_mod.run_subject_worker(
                request, artifact_dir, timeout_s=timeout_s
            )
            cast_responses = info["responses"]
            assert isinstance(cast_responses, list)
            cast_responses.append(response)
            cast_artifacts = info["raw_artifacts"]
            assert isinstance(cast_artifacts, list)
            artifact_stem = response_path.name.removesuffix(".response.json")
            cast_artifacts.extend(
                str(path)
                for path in sorted(response_path.parent.glob(f"{artifact_stem}.*"))
                if path.is_file()
            )
            if not response.complete:
                disabled.add(subject)

    for subject in labels:
        info = subject_info[subject]
        responses = info["responses"]
        assert isinstance(responses, list)
        expected_positions = [
            launch_position
            for session in schedule
            for scheduled_subject, launch_position in session
            if scheduled_subject == subject
        ]
        expected_coordinates = [
            (session_index, launch_position, f"{run_id}:session-{session_index}")
            for session_index, session in enumerate(schedule)
            for scheduled_subject, launch_position in session
            if scheduled_subject == subject
        ]
        run = bench_mod.summarize_workers(
            subject,
            responses,
            inputs,
            expected_workers=workers,
            expected_warmups=warmups,
            expected_repeats=repeats,
            expected_launch_positions=expected_positions,
            expected_session_coordinates=expected_coordinates,
            experiment_id=run_id,
        )
        run["raw_artifacts"] = info["raw_artifacts"]
        if not bool(info["available"]):
            run["error"] = f"backend unavailable: {info['reason']}"
            run["baseline_eligible"] = False
        preflight_issues = info["preflight_issues"]
        if preflight_issues:
            run["provenance_issues"] = list(
                dict.fromkeys([*run.get("provenance_issues", []), *preflight_issues])
            )
            run["baseline_eligible"] = False
        protocol_issues: list[str] = []
        if workers < 5:
            protocol_issues.append("trusted performance requires at least five sessions")
        if warmups < 3:
            protocol_issues.append("trusted performance requires at least three warmups")
        if repeats < 3:
            protocol_issues.append("trusted performance requires at least three repeats")
        if profile:
            protocol_issues.append("profiling is disabled for trusted performance")
        if protocol_issues:
            run["eligibility_issues"] = protocol_issues
            run["baseline_eligible"] = False
        measurement["runs"].append(run)

    write_json_atomic(measurement, run_dir / "summary.json")
    console.print(f"[dim]raw measurement → {run_dir}[/dim]")

    if update:
        failed_runs = [run for run in measurement["runs"] if run.get("error")]
        ineligible_runs = [
            run for run in measurement["runs"] if not run.get("baseline_eligible", False)
        ]
        if failed_runs or ineligible_runs:
            console.print(
                "\n[red]baseline not written: "
                f"{len(failed_runs)} incomplete and {len(ineligible_runs)} "
                "provenance-ineligible subject(s)[/red]"
            )
            raise typer.Exit(1)
        existing = existing_baseline
        if existing and existing.get("artifact_kind") != "baseline-v2":
            console.print(
                "\n[red]baseline not written: schema-v1 baselines are diagnostic-only; "
                "create a fresh v2 baseline artifact instead[/red]"
            )
            raise typer.Exit(1)
        if existing and existing.get("artifact_kind") == "baseline-v2":
            changed_transcripts = bench_mod.transcript_changes(existing, measurement)
            if changed_transcripts and not accept_transcript_changes:
                console.print(
                    "\n[red]baseline not written: transcript changes require "
                    "--accept-transcript-changes --approval-note[/red]"
                )
                raise typer.Exit(1)
            if changed_transcripts:
                summary_entry = next(
                    item
                    for item in existing.get("artifact_manifest", [])
                    if item.get("kind") == "summary"
                )
                measurement["transcript_approval"] = {
                    "note": approval_note,
                    "subjects": sorted({str(item["subject"]) for item in changed_transcripts}),
                    "changed_transcripts": changed_transcripts,
                    "supersedes_baseline_id": existing.get("baseline_id"),
                    "superseded_summary": {
                        "path": summary_entry.get("path"),
                        "sha256": summary_entry.get("sha256"),
                    },
                }
        bench_mod.save_baseline(measurement, baseline_path)
        console.print(f"\n[green]baseline written[/green] → {baseline_path}")
        return

    stored = existing_baseline
    if stored is None:
        raise typer.BadParameter(
            f"No baseline at {baseline_path}. Create one with: stt bench --update"
        )

    if only:
        selected = set(only)
        stored = {
            **stored,
            "runs": [
                run for run in stored.get("runs", []) if run["subject"].split("/", 1)[0] in selected
            ],
        }
    verdict = bench_mod.compare(stored, measurement)
    if verdict.host_note:
        console.print(f"\n[yellow]{verdict.host_note}[/yellow]")

    table = Table(title="Regression check")
    table.add_column("Subject", style="bold")
    table.add_column("Signal")
    table.add_column("Detail")
    for finding in verdict.findings:
        style = "red" if finding.fatal else "yellow"
        table.add_row(finding.subject, f"[{style}]{finding.signal}[/{style}]", finding.detail)
    if verdict.findings:
        console.print(table)

    measured = len(measurement["runs"])
    if verdict.ok:
        console.print(
            f"\n[green]{measured} subject(s) match the baseline[/green]"
            + (f" · {len(verdict.warnings)} warning(s)" if verdict.warnings else "")
        )
        return
    console.print(f"\n[red]{len(verdict.failures)} regression(s)[/red] against {baseline_path}")
    raise typer.Exit(1)


@app.command("check-encoding")
def check_encoding(
    text: Annotated[list[str], typer.Argument(help="Text, or a path to a .txt/.tsv file")],
) -> None:
    """Report whether Burmese text is Unicode or Zawgyi."""
    for item in text:
        p = Path(item)
        content = p.read_text(encoding="utf-8") if p.is_file() else item
        label = p.name if p.is_file() else (item[:40] + "…" if len(item) > 40 else item)
        console.print(f"{label}: [bold]{describe_encoding(content)}[/bold]")


if __name__ == "__main__":
    app()
