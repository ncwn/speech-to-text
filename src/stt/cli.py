"""Command-line interface.

stt backends                     what is installed and working
stt hardware                     what this machine is and its fastest precision
stt models                       available model names per backend
stt fetch-fleurs                 download Burmese eval audio + references
stt transcribe AUDIO...          run one backend
stt eval RESULTS.jsonl           score a run against references
stt align AUDIO --text FILE      time an existing transcript (subtitles)
stt compare AUDIO...             run every installed backend and compare
"""

from __future__ import annotations

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
from stt.vote import DEFAULT_WEIGHTS, rover

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


def _fmt(value: float | None, spec: str = ".4f", dash: str = "—") -> str:
    if value is None or value != value:  # None or NaN
        return dash
    return format(value, spec)


# --------------------------------------------------------------------- info


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
    from stt.backends.omniasr_gguf import DEFAULT_MODEL as GGUF_DEFAULT
    from stt.backends.omniasr_gguf import MODELS as GGUF_MODELS
    from stt.backends.omniasr_torch import DEFAULT_MODEL as TORCH_DEFAULT

    if backend in (None, "omniasr-gguf"):
        table = Table(title="omniasr-gguf  (Metal GPU)")
        table.add_column("Model", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Long audio")
        for key, spec in GGUF_MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == GGUF_DEFAULT else key
            table.add_row(
                label,
                f"{spec.approx_mb} MB",
                "unlimited" if spec.unlimited else "chunked",
            )
        console.print(table)

    if backend in (None, "omniasr-torch"):
        table = Table(title="omniasr-torch  (CPU)")
        table.add_column("Model card", style="bold")
        table.add_column("Download", justify="right")
        table.add_column("Long audio")
        rows = [
            ("omniASR_LLM_Unlimited_300M_v2", "6.5 GB", "unlimited"),
            ("omniASR_LLM_Unlimited_1B_v2", "9.1 GB", "unlimited"),
            ("omniASR_LLM_Unlimited_3B_v2", "17.5 GB", "unlimited"),
            ("omniASR_LLM_Unlimited_7B_v2", "31.2 GB", "unlimited"),
            ("omniASR_LLM_7B_v2", "31.2 GB", "40 s max"),
            ("omniASR_CTC_7B_v2", "~30 GB", "40 s max"),
        ]
        for card, size, limit in rows:
            label = f"{card}  [dim](default)[/dim]" if card == TORCH_DEFAULT else card
            table.add_row(label, size, limit)
        console.print(table)
        console.print(
            "[dim]Any card from facebookresearch/omnilingual-asr works; "
            "these are the common ones.[/dim]"
        )

    if backend in (None, "hf"):
        from stt.backends.transformers_asr import DEFAULT_MODEL as HF_DEFAULT
        from stt.backends.transformers_asr import MODELS as HF_MODELS

        table = Table(title="hf  (Metal GPU via transformers)")
        table.add_column("Model", style="bold")
        table.add_column("Download", justify="right")
        table.add_column("Family")
        table.add_column("Notes")
        for key, spec in HF_MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == HF_DEFAULT else key
            table.add_row(label, f"{spec.approx_mb} MB", spec.family, spec.note)
        console.print(table)
        console.print("[dim]MMS and SeamlessM4T weights are CC-BY-NC-4.0.[/dim]")

    if backend in (None, "dolphin"):
        from stt.backends.dolphin import DEFAULT_MODEL as DOLPHIN_DEFAULT
        from stt.backends.dolphin import MODELS as DOLPHIN_MODELS

        table = Table(title="dolphin  (CPU)")
        table.add_column("Model", style="bold")
        table.add_column("Params", justify="right")
        table.add_column("Download", justify="right")
        for key, spec in DOLPHIN_MODELS.items():
            label = f"{key}  [dim](default)[/dim]" if key == DOLPHIN_DEFAULT else key
            table.add_row(label, f"{spec.params_m}M", f"{spec.approx_mb} MB")
        console.print(table)
        console.print("[dim]Only base and small were publicly released.[/dim]")


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
        "[dim]Thread counts are left to macOS and to each runtime: measured on this "
        "stack, 4, 8 and 12 threads all give the same RTF.[/dim]"
    )


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

    Only touches results that need it, so a backend with native timestamps
    keeps its own — they are measured, whereas these are inferred.
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

    Results usually record the *converted* file, which `stt.audio.to_16k_mono`
    names ``<parent>__<stem>.16k.wav`` to keep same-named files in different
    folders apart. So the original path has to be matched against that
    derived name as well as against itself.
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
        typer.Option("--weight", "-w", help="model=value, repeatable. Default: measured ranking"),
    ] = None,
) -> None:
    """Combine several transcription runs by weighted per-character vote.

    Different systems fail on different words, so voting recovers accuracy that
    no single model reaches. Measured: FLEURS 0.1017 -> 0.0930, held-out audio
    0.0857 -> 0.0714. The first run given is the pivot and should be your best
    model — voting can only correct characters the pivot proposed.
    """
    if len(runs) < 2:
        raise typer.BadParameter("need at least two runs to vote between")

    weights = dict(DEFAULT_WEIGHTS)
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

    ``stt compare`` runs every installed backend at its default model, and the
    omniasr-torch default is a 31 GB download. Starting that unannounced is a
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
) -> None:
    """Run every installed backend over the same audio and tabulate the results."""
    files = _prepare(audio, limit, convert=True)
    refs = load_references(reference) if reference else None

    table = Table(title="Backend comparison")
    table.add_column("Backend", style="bold")
    table.add_column("Model")
    table.add_column("CER", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Failed", justify="right")

    ran = 0
    for name, cls in all_backends().items():
        if only and name not in only:
            continue
        ok, reason = cls.is_available()
        if not ok:
            console.print(f"[yellow]skipping {name}[/yellow] — {reason}")
            continue

        if not yes and not _confirm_download(cls):
            console.print(f"[yellow]skipping {name}[/yellow] — download declined")
            continue

        results = _run_backend(name, None, files, language, 1, {})
        write_jsonl(results, output_dir / f"{name}.jsonl")
        ran += 1

        cer = "—"
        if refs:
            cer = _fmt(score_results(results, refs).cer)
        failed = sum(1 for r in results if r.error)
        model = results[0].model if results else "?"
        table.add_row(name, model, cer, _fmt(mean_rtf(results), ".2f"), str(failed))

    if not ran:
        console.print("[red]No backends available.[/red] Run `stt backends` to see why.")
        raise typer.Exit(1)

    console.print()
    console.print(table)
    if not refs:
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
