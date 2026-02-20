"""Container entrypoint for the SkillForge Ollama sample."""

from pathlib import Path
import importlib.util


def _load_sample_module():
    sample_path = Path(__file__).resolve().parent.parent / "sample_skillforge_langgraph_ollama.py"
    spec = importlib.util.spec_from_file_location("sample_skillforge_langgraph_ollama", sample_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sample module from {sample_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sample_module = _load_sample_module()
    sample_module.run_skillforge_workflow()


if __name__ == "__main__":
    main()
