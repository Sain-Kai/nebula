import subprocess
import sys


def test_cli_demo():
    res = subprocess.run([sys.executable, "-m", "nebula.cli", "demo", "--prompt", "Hi"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Tokenization" in res.stdout


def test_cli_deliberate():
    res = subprocess.run([sys.executable, "-m", "nebula.cli", "deliberate", "--prompt", "Because test", "--max-tokens", "4"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Cognitive Deliberation" in res.stdout


def test_cli_sae():
    res = subprocess.run([sys.executable, "-m", "nebula.cli", "sae", "--prompt", "Test", "--layer", "1", "--k", "8"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Sparse Autoencoder" in res.stdout


def test_cli_speculate():
    res = subprocess.run([sys.executable, "-m", "nebula.cli", "speculate", "--prefix", "Test", "--gamma", "2"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Speculative Rejection" in res.stdout
