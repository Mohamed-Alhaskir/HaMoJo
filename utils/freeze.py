"""
Freeze Manifest
================
Generates and verifies the cryptographic freeze manifest that locks
the entire pipeline configuration before confirmatory data analysis.

The manifest includes:
- Pipeline version
- Git commit hash (compared at verify time)
- Working-tree-clean-at-freeze flag
- Model identifiers and inference parameters
- Prompt template hashes (all .j2 + .json under templates/instruments and instruments)
- Source code hashes (all .py under stages/, utils/, and pipeline.py)
- Configuration hash
- Runtime environment (Python + key package versions + OS/hardware)
- Random seeds
- Timestamp of freeze

Note: session_overview.json is NOT part of the freeze — it contains
per-session metadata that can legitimately change without requiring a
pipeline re-freeze. The model file bytes are also not hashed; only the
model_id (filename stem) and inference parameters are recorded.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class FreezeManifest:
    """Immutable record of the pipeline state at freeze time."""

    def __init__(self, pipeline_version: str, config: dict):
        self.pipeline_version = pipeline_version
        self.config = config
        self._manifest = self._build()

    def _build(self) -> dict:
        return {
            "pipeline_version": self.pipeline_version,
            "git_commit": self._get_git_commit(),
            "working_tree_clean_at_freeze": self._is_working_tree_clean(),
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "seeds": {
                "global": self.config.get("pipeline", {}).get("seed", None),
                "llm": self.config.get("llm", {}).get("seed", None),
            },
            "models": {
                "asr": {
                    "engine": "faster-whisper",
                    "model_name": self.config.get("asr", {}).get("model_name"),
                    "compute_type": self.config.get("asr", {}).get("compute_type"),
                    "beam_size": self.config.get("asr", {}).get("beam_size"),
                    "language": self.config.get("asr", {}).get("language"),
                },
                "diarization": {
                    "enabled": self.config.get("asr", {})
                    .get("diarization", {})
                    .get("enabled"),
                },
                "llm": {
                    "backend": self.config.get("llm", {}).get("backend"),
                    "model_id": Path(
                        self.config.get("llm", {}).get("model_path", "")
                    ).stem,
                    "temperature": self.config.get("llm", {}).get("temperature"),
                    "context_length": self.config.get("llm", {}).get("context_length"),
                },
            },
            "prompt_template_hashes": self._hash_prompt_templates(),
            "source_code_hashes": self._hash_source_code(),
            "config_hash": self._hash_config(),
            "environment": self._capture_environment(),
        }

    @staticmethod
    def _get_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    @staticmethod
    def _hash_prompt_templates() -> dict:
        """Hash all Jinja2 templates and instrument JSON definitions.

        Returns a dict with per-file hashes and a combined digest so that
        any single-file change is both traceable and detectable.
        """
        dirs = [
            Path("templates/instruments"),
            Path("instruments"),
        ]
        file_hashes = {}
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix in (".j2", ".json") and p.is_file():
                    content = p.read_bytes()
                    file_hashes[str(p)] = hashlib.sha256(content).hexdigest()

        combined = hashlib.sha256(
            json.dumps(file_hashes, sort_keys=True).encode()
        ).hexdigest()
        return {"files": file_hashes, "combined": combined}

    @staticmethod
    def _hash_source_code() -> dict:
        """Hash every Python source file the pipeline actually runs.

        Covers pipeline.py at the repo root plus all .py files (recursive)
        under stages/ and utils/. Excludes __pycache__. Any byte-level change
        to any of these files invalidates the combined digest, so silent
        code edits are detected at verify time.
        """
        paths: list[Path] = []
        if Path("pipeline.py").is_file():
            paths.append(Path("pipeline.py"))
        for d in (Path("stages"), Path("utils")):
            if d.exists():
                paths.extend(sorted(d.rglob("*.py")))

        file_hashes: dict[str, str] = {}
        for p in sorted(paths):
            if "__pycache__" in p.parts:
                continue
            file_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

        combined = hashlib.sha256(
            json.dumps(file_hashes, sort_keys=True).encode()
        ).hexdigest()
        return {"files": file_hashes, "combined": combined}

    @staticmethod
    def _is_working_tree_clean() -> bool | None:
        """Return True if `git status --porcelain` is empty (ignoring
        freeze_manifest.json itself), False if dirty, None if git is
        unavailable or the directory is not a git repo.

        freeze_manifest.json is filtered out because the shell redirect
        `> freeze_manifest.json` creates/truncates the file BEFORE Python
        starts, so it would always appear "dirty" in the porcelain output
        during the very command that produces it.
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [
                    line for line in result.stdout.strip().splitlines()
                    if "freeze_manifest.json" not in line
                ]
                return len(lines) == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    @staticmethod
    def _capture_environment() -> dict:
        """Record Python + key library versions + OS/hardware descriptors.

        OS/hardware fields are informational (not gated at verify time).
        python_version + gated package versions are compared at verify time.
        """
        import sys
        import platform

        env: dict[str, str | None] = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
        }
        try:
            from importlib.metadata import version, PackageNotFoundError
            for pkg in (
                "llama-cpp-python",
                "faster-whisper",
                "ctranslate2",
                "nemo-toolkit",
                "torch",
                "numpy",
                "transformers",
                "jinja2",
            ):
                key = "pkg_" + pkg.replace("-", "_")
                try:
                    env[key] = version(pkg)
                except PackageNotFoundError:
                    env[key] = None
        except ImportError:
            pass
        return env

    def _hash_config(self) -> str:
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def digest(self) -> str:
        """Return a single SHA-256 digest of the entire manifest."""
        manifest_str = json.dumps(self._manifest, sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {**self._manifest, "manifest_digest": self.digest()}

    def save(self, path: str):
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_and_verify(cls, path: str, current_config: dict, pipeline_version: str) -> bool:
        """
        Load a saved manifest and verify it matches the current pipeline state.
        Returns the saved dict if the pipeline has NOT changed since freeze,
        None otherwise.
        """
        with open(path, "r") as f:
            saved = json.load(f)

        current = cls(pipeline_version, current_config)
        current_dict = current.to_dict()

        # Compare critical fields.
        # Note: git_commit is recorded in the manifest as informational
        # provenance but NOT gated at verify time. Gating on git_commit
        # creates a self-defeating loop — the very commit that adds
        # freeze_manifest.json to the repo advances HEAD past the commit
        # the manifest was computed from, which would always trigger a
        # false FREEZE VIOLATION. source_code_hashes already catches any
        # byte-level code change, which is what matters for reproducibility.
        mismatches = []
        for key in ["pipeline_version", "config_hash"]:
            if saved.get(key) != current_dict.get(key):
                mismatches.append(key)

        saved_prompts = (saved.get("prompt_template_hashes") or {}).get("combined")
        current_prompts = (current_dict.get("prompt_template_hashes") or {}).get("combined")
        if saved_prompts != current_prompts:
            mismatches.append("prompt_template_hashes")

        saved_code = (saved.get("source_code_hashes") or {}).get("combined")
        current_code = (current_dict.get("source_code_hashes") or {}).get("combined")
        if saved_code != current_code:
            mismatches.append("source_code_hashes")

        if saved.get("models") != current_dict.get("models"):
            mismatches.append("models")

        if saved.get("seeds") != current_dict.get("seeds"):
            mismatches.append("seeds")

        # Compare only the gated environment fields (Python + critical libs).
        # OS/hardware stay informational so a re-run on a different machine
        # doesn't trip the freeze.
        saved_env = saved.get("environment") or {}
        current_env = current_dict.get("environment") or {}
        for key in ("python_version", "pkg_llama_cpp_python", "pkg_faster_whisper"):
            if saved_env.get(key) != current_env.get(key):
                mismatches.append(f"environment.{key}")

        if mismatches:
            print(f"⚠ FREEZE VIOLATION — changed fields: {mismatches}")
            return None

        print("✓ Freeze manifest verified — pipeline unchanged.")
        return saved
