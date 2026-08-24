"""Command-line interface.

stt backends                     what is installed and working
stt hardware                     what this machine is and its fastest precision
stt models                       available model names per backend
stt fetch-fleurs                 download Burmese eval audio + references
stt transcribe AUDIO...          run one backend
stt eval RESULTS.jsonl           score a run against references
stt align AUDIO --text FILE      time an existing transcript (subtitles)
stt compare AUDIO...             compare cached models or selected backends
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from stt import audio as audio_mod
from stt.burmese import NormalizeOptions, describe_encoding
from stt.evaluate import load_references, mean_rtf, score_results
from stt.registry import all_backends, get_backend
from stt.results import (
    TranscriptionResult,
    read_jsonl,
    write_jsonl,
    write_srt,
    write_text,
    write_vtt,
)
from stt.telemetry import describe_host, measure
from stt.vote import rover

app = typer.Typer(
    name="stt",
    help="Burmese-focused speech-to-text testing harness.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

#: omniASR's code for Burmese. Both backends accept it.
BURMESE = "mya_Mymr"

DEFAULT_CACHE = Path("data/.converted")
DEFAULT_OUTPUT_DIR = Path("outputs")


def _configure_project_model_caches(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[2] / ".cache"
    for variable, relative, alternatives in (
        ("CRISPASR_CACHE_DIR", "crispasr", ()),
        ("DOLPHIN_CACHE_DIR", "dolphin", ()),
        ("FAIRSEQ2_CACHE_DIR", "fairseq2/assets", ("XDG_CACHE_HOME",)),
        ("HF_HUB_CACHE", "huggingface/hub", ("HF_HOME", "XDG_CACHE_HOME")),
    ):
        path = root / relative
        if path.is_dir() and not any(name in os.environ for name in (variable, *alternatives)):
            os.environ[variable] = str(path)


_configure_project_model_caches()


def _fmt(value: float | None, spec: str = ".4f", dash: str = "—") -> str:
    if value is None or value != value:  # None or NaN
        return dash
    return format(value, spec)


# --------------------------------------------------------------------- info


def _offline_status(cls, model: str) -> str:
    try:
        cached = cls(model).weights_cached()
    except (ImportError, RuntimeError):
        cached = None
    if cached is None:
        return "[yellow]unknown[/yellow]"
    return "[green]yes[/green]" if cached else "[dim]no[/dim]"


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
    download: Annotated[
        str | None, typer.Option("--download", help="Download one model into its upstream cache")
    ] = None,
) -> None:
    """List selectable models, or download one model into its upstream cache."""
    from stt.backends.omniasr_gguf import DEFAULT_MODEL as GGUF_DEFAULT
    from stt.backends.omniasr_gguf import MODELS as GGUF_MODELS
    from stt.backends.omniasr_torch import DEFAULT_MODEL as TORCH_DEFAULT
    from stt.backends.omniasr_torch import MODELS as TORCH_MODELS

    known = all_backends()
    if backend is not None and backend not in known:
        raise typer.BadParameter(
            f"Unknown backend {backend!r}. Available: {', '.join(known)}", param_hint="--backend"
        )
    if download is not None and backend is None:
        raise typer.BadParameter("--backend is required with --download", param_hint="--backend")

    if backend in (None, "omniasr-gguf"):
        table = Table(title="omniasr-gguf  (runtime-selected device)")
        table.add_column("Model", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Offline")
        table.add_column("Long audio")
        for key, spec in GGUF_MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == GGUF_DEFAULT else key
            table.add_row(
                label,
                f"{spec.approx_mb} MB",
                _offline_status(known["omniasr-gguf"], key),
                "unlimited" if spec.unlimited else "chunked",
            )
        console.print(table)

    if backend in (None, "omniasr-torch"):
        table = Table(title="omniasr-torch  (auto device)")
        table.add_column("Model card", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Offline")
        table.add_column("Long audio")
        for card, spec in TORCH_MODELS.items():
            label = f"{card}  [dim](default)[/dim]" if card == TORCH_DEFAULT else card
            table.add_row(
                label,
                f"{spec.approx_mb} MB",
                _offline_status(known["omniasr-torch"], card),
                "unlimited" if spec.unlimited else "40 s max",
            )
        console.print(table)

    if backend in (None, "hf"):
        from stt.backends.transformers_asr import DEFAULT_MODEL as HF_DEFAULT
        from stt.backends.transformers_asr import MODELS as HF_MODELS

        table = Table(title="hf  (auto device)")
        table.add_column("Model", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Offline")
        table.add_column("Family")
        table.add_column("Notes")
        for key, spec in HF_MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == HF_DEFAULT else key
            table.add_row(
                label,
                f"{spec.approx_mb} MB",
                _offline_status(known["hf"], key),
                spec.family,
                spec.note,
            )
        console.print(table)
        console.print("[dim]MMS and SeamlessM4T weights are CC-BY-NC-4.0.[/dim]")

    if backend in (None, "dolphin"):
        from stt.backends.dolphin import DEFAULT_MODEL as DOLPHIN_DEFAULT
        from stt.backends.dolphin import MODELS as DOLPHIN_MODELS

        table = Table(title="dolphin  (CPU default)")
        table.add_column("Model", style="bold")
        table.add_column("Params", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Offline")
        for key, spec in DOLPHIN_MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == DOLPHIN_DEFAULT else key
            table.add_row(
                label,
                f"{spec.params_m}M",
                f"{spec.approx_mb} MB",
                _offline_status(known["dolphin"], key),
            )
        console.print(table)
        console.print("[dim]This adapter exposes the base and small cards.[/dim]")

    if download is not None:
        assert backend is not None
        cls = known[backend]
        ok, reason = cls.is_available()
        if not ok:
            raise typer.BadParameter(
                f"Backend {backend!r} is not available: {reason}. Install with: {cls.install_hint}",
                param_hint="--backend",
            )
        try:
            selected = cls(download)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--download") from exc
        console.print(f"Downloading [bold]{backend}[/bold] · [cyan]{download}[/cyan]")
        selected.download_weights()
        console.print(f"[green]cached[/green] · {backend} · {download}")


@app.command()
def hardware(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-run the dtype probe instead of using the cache")
    ] = False,
) -> None:
    """Show what this machine is, and which precision it runs fastest.

    Capabilities are read from the machine; dtype probe results are cached.
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
    console.print("[dim]Thread counts are left to the operating system and runtime.[/dim]")


# ------------------------------------------------------------------- data


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


# -------------------------------------------------------------- transcribe


def _report_resources(results: list[TranscriptionResult]) -> None:
    """Summarise what the run cost, beyond wall-clock time."""
    usages = [r.resources for r in results if r.resources]
    if not usages:
        return
    cores = [u.cpu_utilization for u in usages if u.cpu_utilization is not None]
    gpu = [u.gpu_mb for u in usages if u.gpu_mb is not None]
    line = f"[dim]CPU {sum(cores) / len(cores):.1f} cores busy" if cores else "[dim]CPU —"
    line += f" · peak RSS {max(u.peak_rss_mb for u in usages):.0f} MB"
    if gpu:
        line += f" · GPU {max(gpu):.0f} MB"
    console.print(line + "[/dim]")


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


def _add_alignment(results: list[TranscriptionResult], device: str = "cpu") -> None:
    """Fill in timings for results whose backend could not supply any.

    Existing native or chunk timings remain unchanged; only missing timings are
    inferred.
    """
    from stt.align import align, load_aligner

    pending = [r for r in results if not r.error and r.text.strip() and not r.segments]
    if not pending:
        return

    with console.status(f"Aligning {len(pending)} transcript(s)…"):
        aligner = load_aligner(device)
        for r in pending:
            try:
                r.segments = align(r.text, Path(r.audio_path), aligner=aligner) or None
            except Exception as exc:  # noqa: BLE001 - alignment is best-effort
                r.metadata["align_error"] = f"{type(exc).__name__}: {exc}"
                console.print(f"[yellow]align failed for {Path(r.audio_path).name}: {exc}[/yellow]")


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


def _prepare(paths: list[Path], limit: int, convert: bool) -> list[Path]:
    files = audio_mod.find_audio(paths)
    if not files:
        raise typer.BadParameter("No audio files found")
    if limit:
        files = files[:limit]
    if convert:
        files = [audio_mod.to_16k_mono(f, DEFAULT_CACHE) for f in files]
    return files


def _run_backend(
    backend_name: str,
    model: str | None,
    files: list[Path],
    language: str | None,
    batch_size: int,
    options: dict,
) -> list[TranscriptionResult]:
    cls = get_backend(backend_name)
    ok, reason = cls.is_available()
    if not ok:
        raise typer.BadParameter(
            f"Backend {backend_name!r} is not available: {reason}. Install with: {cls.install_hint}"
        )

    kwargs = {k: v for k, v in options.items() if v is not None}
    unsupported = kwargs.keys() - cls.supported_options
    if unsupported:
        option = sorted(unsupported)[0]
        flag = "--threads" if option == "n_threads" else f"--{option.replace('_', '-')}"
        raise typer.BadParameter(
            f"{flag} is not supported by backend {backend_name!r}", param_hint=flag
        )
    if (device := kwargs.get("device")) not in (None, "auto", "cpu", "mps", "cuda"):
        raise typer.BadParameter(
            f"Unknown device {device!r}. Choose from: auto, cpu, mps, cuda",
            param_hint="--device",
        )
    if (dtype := kwargs.get("dtype")) not in (
        None,
        "auto",
        "float32",
        "fp32",
        "float16",
        "fp16",
        "bfloat16",
        "bf16",
    ):
        raise typer.BadParameter(
            f"Unknown dtype {dtype!r}. Choose from: auto, float32, float16, bfloat16",
            param_hint="--dtype",
        )
    instance = cls(model, **kwargs) if model else cls(**kwargs)

    console.print(
        f"[bold]{backend_name}[/bold] · model=[cyan]{instance.model}[/cyan] · "
        f"{len(files)} file(s) · lang={language or 'auto'}"
    )
    with console.status("Loading model…"):
        instance.load()

    results: list[TranscriptionResult] = []
    # Stamped on every record so a timing stays interpretable later: the same
    # model on a different machine is a different number.
    host = describe_host()
    with typer.progressbar(files, label="Transcribing") as bar:
        for f in bar:
            # Measured here rather than inside each backend: this is the one
            # place every backend passes through, so all of them are
            # instrumented identically and none can forget to be.
            with measure() as usage:
                batch = instance.transcribe([f], language=language, batch_size=batch_size)
            for r in batch:
                r.resources = usage[0]
                r.metadata.setdefault("host", host)
            results.extend(batch)
    instance.unload()
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
        str | None,
        typer.Option(help="omniasr-torch/hf/dolphin device: auto, cpu, mps, cuda"),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option(help="omniasr-torch/hf dtype: auto, float32, float16, bfloat16"),
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
        bool,
        typer.Option("--verbose", "-v", help="omniasr-gguf/dolphin: show native logs"),
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

    if do_align or srt or vtt:
        _add_alignment(results)

    out = output or DEFAULT_OUTPUT_DIR / f"{backend}.jsonl"
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

    rtf = mean_rtf(results)
    failed = sum(1 for r in results if r.error)
    console.print(
        f"\n[green]{len(results) - failed}/{len(results)} transcribed[/green] · "
        f"mean RTF {_fmt(rtf, '.2f')} → [bold]{out}[/bold]"
    )
    _report_resources(results)


# -------------------------------------------------------------------- align
def _same_audio(recorded: str, audio: Path) -> bool:
    """Whether a stored result refers to ``audio``.

    Current converted files retain the source stem. The parent-prefixed form is
    also accepted for JSONL written by older versions.
    """
    stem = Path(recorded).stem.removesuffix(".16k")
    return stem in {audio.stem, f"{audio.parent.name}__{audio.stem}"}


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

    if (text is None) == (results is None):
        raise typer.BadParameter("Pass exactly one of --text or --results")

    if text is not None:
        transcript = text.read_text(encoding="utf-8")
    else:
        assert results is not None
        loaded = read_jsonl(results)
        matching = [r for r in loaded if _same_audio(r.audio_path, audio)]
        if not matching:
            known = ", ".join(sorted({Path(r.audio_path).stem for r in loaded})[:3])
            raise typer.BadParameter(f"No result in {results} for {audio.stem!r}. Found: {known}")
        transcript = matching[0].text

    converted = audio_mod.to_16k_mono(audio, DEFAULT_CACHE)
    with console.status("Aligning…"):
        segments = align_text(transcript, converted, device=device)

    if not segments:
        console.print("[red]Alignment produced no segments.[/red]")
        raise typer.Exit(1)

    result = TranscriptionResult(
        audio_path=str(audio),
        text=transcript,
        backend="align",
        model="mms-1b-all",
        audio_duration_s=audio_mod.duration_of(converted),
        segments=segments,
    )
    stem = output or DEFAULT_OUTPUT_DIR / f"{audio.stem}"
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


# --------------------------------------------------------------------- eval


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

    console.print(
        f"\n[bold]{score.backend}[/bold] · {score.model}\n"
        f"  CER   [bold cyan]{_fmt(score.cer)}[/bold cyan]   "
        f"({len(score.scored)} clip(s) scored"
        + (f", [red]{score.n_failed} failed[/red]" if score.n_failed else "")
        + ")\n"
        f"  WER   {_fmt(score.wer)}   [dim](not meaningful for Burmese)[/dim]\n"
        f"  RTF   {_fmt(mean_rtf(results), '.2f')}"
    )


# --------------------------------------------------------------------- vote


@app.command()
def vote(
    runs: Annotated[
        list[Path],
        typer.Argument(help="JSONL runs to combine. Put your most accurate model FIRST."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the result")],
    weight: Annotated[
        list[str] | None,
        typer.Option("--weight", "-w", help="model=value, repeatable. Default: 1.0"),
    ] = None,
) -> None:
    """Combine several transcription runs by weighted per-character vote.

    The first run is the pivot and should be the strongest input because voting
    can only correct characters the pivot proposed.
    """
    if len(runs) < 2:
        raise typer.BadParameter("need at least two runs to vote between")

    weights: dict[str, float] = {}
    for item in weight or []:
        name, _, value = item.partition("=")
        if not value:
            raise typer.BadParameter(f"--weight expects model=value, got {item!r}")
        weights[name.strip()] = float(value)

    # Group every run by audio file, keeping the order the runs were given so
    # runs[0] stays the pivot.
    by_audio: dict[str, dict[str, TranscriptionResult]] = {}
    labels: list[str] = []
    for path in runs:
        results = read_jsonl(path)
        label = results[0].model if results else path.stem
        while label in labels:  # two runs of the same model: keep both distinct
            label += "'"
        labels.append(label)
        for r in results:
            by_audio.setdefault(r.audio_path, {})[label] = r

    pivot = labels[0]
    combined: list[TranscriptionResult] = []
    skipped = 0
    for audio_path, per_model in by_audio.items():
        if pivot not in per_model:
            skipped += 1
            continue
        base = per_model[pivot]
        texts = {name: (r.text or "") for name, r in per_model.items()}
        combined.append(
            TranscriptionResult(
                audio_path=audio_path,
                text=rover(texts, pivot, weights),
                backend="vote",
                model="+".join(sorted(per_model)),
                language=base.language,
                elapsed_s=sum(r.elapsed_s or 0.0 for r in per_model.values()) or None,
                audio_duration_s=base.audio_duration_s,
                metadata={"pivot": pivot, "voters": sorted(per_model)},
            )
        )

    write_jsonl(combined, output)
    console.print(
        f"voted {len(combined)} file(s) across {len(labels)} run(s) "
        f"[dim](pivot: {pivot})[/dim] → {output}"
        + (
            f"\n[yellow]{skipped} file(s) skipped: missing from the pivot run[/yellow]"
            if skipped
            else ""
        )
    )


# ------------------------------------------------------------------ compare


def _confirm_download(cls) -> bool:
    """Ask before a backend's default model pulls a large checkpoint.

    An explicitly selected backend may still need its default model. Starting a
    multi-gigabyte download unannounced is a
    nasty surprise, so anything over a gigabyte that is not known to be cached
    gets a prompt.
    """
    try:
        probe = cls()
    except Exception:  # noqa: BLE001 - construction may need arguments we lack
        return True

    if probe.weights_cached() is True:
        return True
    size_mb = probe.estimated_download_mb()
    if size_mb is None or size_mb < 1024:
        return True

    return typer.confirm(
        f"{cls.name}: {probe.model} may need to download ~{size_mb / 1024:.1f} GB. Continue?",
        default=False,
    )


def _model_names_by_backend() -> dict[str, tuple[str, ...]]:
    from stt.backends.dolphin import MODELS as DOLPHIN_MODELS
    from stt.backends.omniasr_gguf import MODELS as GGUF_MODELS
    from stt.backends.omniasr_torch import MODELS as TORCH_MODELS
    from stt.backends.transformers_asr import MODELS as HF_MODELS

    return {
        "omniasr-gguf": tuple(GGUF_MODELS),
        "omniasr-torch": tuple(TORCH_MODELS),
        "hf": tuple(HF_MODELS),
        "dolphin": tuple(DOLPHIN_MODELS),
    }


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
    all_cached_models: Annotated[
        bool,
        typer.Option(
            "--all-cached-models",
            help="Run every model whose existing cache can be verified",
        ),
    ] = False,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_OUTPUT_DIR,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not prompt before large downloads")
    ] = False,
) -> None:
    """Compare cached models, or explicitly selected backend defaults."""
    if all_cached_models and only is not None:
        raise typer.BadParameter(
            "--all-cached-models cannot be combined with --only",
            param_hint="--all-cached-models",
        )

    files = _prepare(audio, limit, convert=True)
    refs = load_references(reference) if reference else None

    table = Table(title="Backend comparison")
    table.add_column("Backend", style="bold")
    table.add_column("Model")
    table.add_column("CER", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Failed", justify="right")

    known = all_backends()
    targets = (
        [
            (name, model)
            for name, models in _model_names_by_backend().items()
            if name in known
            for model in models
        ]
        if all_cached_models
        else [(name, None) for name in known]
    )

    skipped_backends: set[str] = set()
    ran = 0
    for name, selected_model in targets:
        if name in skipped_backends:
            continue
        cls = known[name]
        if only is not None and name not in only:
            continue
        ok, reason = cls.is_available()
        if not ok:
            console.print(f"[yellow]skipping {name}[/yellow] — {reason}")
            skipped_backends.add(name)
            continue

        if only is None:
            probe = cls(selected_model) if selected_model else cls()
            cached = probe.weights_cached()
            if cached is None:
                instruction = (
                    f"benchmark it explicitly with `stt transcribe -b {name} -m {probe.model}`"
                    if all_cached_models
                    else f"select it explicitly with `--only {name}`"
                )
                console.print(
                    f"[yellow]skipping {name} · {probe.model}[/yellow] — "
                    f"cache status cannot be verified; {instruction}"
                )
                if all_cached_models:
                    skipped_backends.add(name)
                continue
            if not cached:
                label = f"{name} · {probe.model}" if all_cached_models else name
                console.print(
                    f"[yellow]skipping {label}[/yellow] — run "
                    f"`stt models --backend {name} --download {probe.model}`; model is not cached"
                )
                continue
        elif not yes and not _confirm_download(cls):
            console.print(f"[yellow]skipping {name}[/yellow] — download declined")
            continue

        options = {"local_files_only": True} if only is None and name == "hf" else {}
        results = _run_backend(name, selected_model, files, language, 1, options)
        suffix = f"{name}--{selected_model}" if selected_model else name
        write_jsonl(results, output_dir / f"{suffix}.jsonl")
        ran += 1

        cer = "—"
        failed = sum(1 for r in results if r.error)
        if refs is not None:
            score = score_results(results, refs)
            failed = score.n_failed
            if not failed:
                cer = _fmt(score.cer)
        model = results[0].model if results else "?"
        table.add_row(name, model, cer, _fmt(mean_rtf(results), ".2f"), str(failed))

    if not ran:
        console.print(
            "[red]No backends ran.[/red] Check installation and model cache status above."
        )
        raise typer.Exit(1)

    console.print()
    console.print(table)
    if refs is None:
        console.print("[dim]Pass --reference to get CER instead of just timings.[/dim]")


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
