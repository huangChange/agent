"""Command line entry point: `gsforge run job.yaml`, `gsforge eval`, `gsforge tools`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from .core.config import load_settings
from .observability.logging import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(job_file: Path, profile: str = typer.Option(None, help="delivery profile (overrides the job's)"),
        no_agent: bool = typer.Option(False, help="deterministic pipeline only"),
        inline: bool = typer.Option(False, help="run tools in-process (debugging)"),
        log_json: bool = typer.Option(False, help="JSON logs for log shippers")) -> None:
    """Run one job described by a YAML file."""
    import yaml

    from .pipeline.factory import build_services
    from .pipeline.ingest import ingest_job
    from .pipeline.orchestrator import Pipeline

    setup_logging(json_format=log_json)
    spec = yaml.safe_load(job_file.read_text(encoding="utf-8"))
    settings = load_settings(profile or spec.get("profile", "default"))
    svc = build_services(settings, inline=inline, use_llm=not no_agent)
    job = ingest_job(spec, svc.store, job_file.parent)

    async def main():
        try:
            return await Pipeline(svc).run(job)
        finally:
            svc.runner.backend.shutdown()

    result = asyncio.run(main())
    typer.echo(json.dumps({"job_id": result.job_id, "grades": result.grades, "cost_usd": round(result.cost_usd, 4),
                           "delivery_dir": str(result.delivery_dir), "warnings": result.warnings}, indent=2))


@app.command("eval")
def eval_synthetic(out: Path = typer.Option(Path("./.gsforge/eval"), help="working directory"),
                   use_agent: bool = typer.Option(False, help="enable the repair agent (calls the Claude API)"),
                   profile: str = "default") -> None:
    """Run the synthetic ground-truth benchmark and print scores."""
    from .evals.benchmark import run_synthetic

    setup_logging()
    settings = load_settings(profile, extra={"storage": {"root": str(out / "store")},
                                             "export": {"out_dir": str(out / "deliveries")},
                                             "runtime": {"workdir": str(out / "work")}})
    typer.echo(json.dumps(asyncio.run(run_synthetic(out, settings, use_llm=use_agent)), indent=2))


@app.command()
def tools() -> None:
    """List registered tools with stage, resource and version."""
    from .tools.registry import load_builtin_tools

    for name, spec in sorted(load_builtin_tools().items()):
        first = spec.description.splitlines()[0] if spec.description else ""
        typer.echo(f"{name:28s} v{spec.version:7s} {spec.resource:4s} {','.join(spec.stages):18s} {first}")


if __name__ == "__main__":
    app()
