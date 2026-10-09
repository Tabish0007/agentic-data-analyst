import os
import subprocess
import sys
import tempfile

def list_files(data_dir: str) -> str:
    if not os.path.isdir(data_dir):
        return f"ERROR: data dir '{data_dir}' not found. Run `python scripts/generate_data.py` first."
    lines = []
    readme = os.path.join(data_dir, "README.md")
    if os.path.exists(readme):
        lines.append("=== DATA DICTIONARY ===")
        lines.append(open(readme).read()[:2000])
        lines.append("")
    for fn in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, fn)
        if not os.path.isfile(p):
            continue
        lines.append(f"=== {fn} ({os.path.getsize(p)} bytes) ===")
        if fn.endswith(".csv"):
            with open(p) as f:
                lines.append("".join(f.readline() for _ in range(6)).rstrip())
    return "\n".join(lines)

def run_python(code: str, data_dir: str, timeout: int = 90) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], cwd=data_dir,
                           capture_output=True, text=True, timeout=timeout)
        out = r.stdout or ""
        if r.stderr:
            out += "\n[stderr]\n" + r.stderr
        if r.returncode != 0:
            out = f"ERROR (exit {r.returncode}):\n" + out
        return out[:8000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: execution timed out (90s). Break the computation into smaller steps."
    finally:
        os.unlink(path)
