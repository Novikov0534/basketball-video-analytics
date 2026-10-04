"""Regression for Colab's inherited MPLBACKEND in an isolated interpreter."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ColabBackendTest(unittest.TestCase):
    def test_cli_ignores_notebook_backend_before_matplotlib_import(self):
        # --help exercises CLI initialization without requiring a GPU or model.
        code = """
import sys
from basketball_cv.cli import main
sys.argv = ['basketball-cv', '--help']
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
import matplotlib
import matplotlib.pyplot as plt
assert matplotlib.get_backend().lower() == 'agg'
fig = plt.figure()
fig.canvas.draw()
plt.close(fig)
print('HEADLESS_BACKEND_OK')
"""
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=ROOT,
            env=dict(os.environ, MPLBACKEND='module://matplotlib_inline.backend_inline'),
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('HEADLESS_BACKEND_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
