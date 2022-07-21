import subprocess
from pathlib import Path

def get_current_code_version():
    return subprocess.check_output(["git", "describe", "--always"],
                                   cwd=Path(__file__).resolve().parent).strip().decode()

def model_name_version(model):
    name = model.__class__.__name__
    version = get_current_code_version()
    return name, version