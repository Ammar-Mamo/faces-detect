from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SMOKE_TEST = ROOT / "smoke_test.py"
DIST_DIR = ROOT / "build" / "smoke_test"


def build_command() -> list[str]:
    python_exe = shutil.which("uv") or shutil.which("python") or sys.executable
    command = [
        python_exe,
    ]

    if Path(python_exe).name.lower() == "uv.exe" or Path(python_exe).name.lower() == "uv":
        command.extend(
            [
                "run",
                "--group",
                "build",
                "python",
                "-m",
                "nuitka",
            ]
        )
    else:
        command.extend(["-m", "nuitka"])

    command.extend(
        [
            "--standalone",
            "--assume-yes-for-downloads",
            "--enable-plugin=pyside6",
            "--include-package=insightface",
            "--include-package=faiss",
            "--include-package=onnxruntime",
            "--include-package-data=onnxruntime",
            "--nofollow-import-to=matplotlib",
            "--nofollow-import-to=pandas",
            "--output-dir=%s" % DIST_DIR,
            "--remove-output",
            "--company-name=MamoTech",
            "--product-name=Detect Smoke Test",
            "--file-version=0.1.0.0",
            "--product-version=0.1.0",
            "--windows-console-mode=force",
            str(SMOKE_TEST),
        ]
    )
    return command


def main() -> int:
    if not SMOKE_TEST.exists():
        print(f"Missing input file: {SMOKE_TEST}", file=sys.stderr)
        return 1

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("NUITKA_ASSUME_YES_FOR_DOWNLOADS", "1")

    command = build_command()
    print("Running build command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
