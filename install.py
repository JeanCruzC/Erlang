"""Install required packages and generate example scripts."""

import subprocess
import sys
from pathlib import Path


def install_requirements(requirements_file: str = "requirements.txt") -> None:
    """Install packages listed in the given requirements file using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])


def create_examples() -> None:
    """Create example scripts demonstrating Erlang calculator workflows."""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    basic_example = examples_dir / "basic_usage.py"
    if not basic_example.exists():
        basic_example.write_text(
            """import erlang_examples as ex\n\nex.run_basic_workflow()\n"""
        )
    print(f"Created example script: {basic_example}")


if __name__ == "__main__":
    install_requirements()
    create_examples()
