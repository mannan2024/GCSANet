"""Figure export helpers used by the training and ablation scripts."""

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path


_MATPLOTLIB_SAVE_SNIPPET = """
import json
import pickle
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(sys.argv[1], 'rb') as f:
    fig = pickle.load(f)

save_kwargs = json.loads(sys.argv[3])
fig.savefig(sys.argv[2], **save_kwargs)
plt.close(fig)
"""

def _short_error_message(result):
    stderr = (result.stderr or '').strip()
    stdout = (result.stdout or '').strip()
    detail = stderr or stdout
    if not detail:
        return f'process exited with code {result.returncode}'
    return detail.splitlines()[-1]


def _run_python_snippet(snippet, args, timeout_seconds):
    try:
        result = subprocess.run(
            [sys.executable, '-c', snippet, *args],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f'timed out after {timeout_seconds}s'

    if result.returncode != 0:
        return False, _short_error_message(result)

    return True, None


def save_matplotlib_bundle(fig, png_path, svg_path=None, png_dpi=600, timeout_seconds=120, **savefig_kwargs):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if svg_path is not None:
        svg_path = Path(svg_path)
        svg_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_pickle_path = Path(tmp_dir) / 'figure.pkl'
        with fig_pickle_path.open('wb') as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)

        png_kwargs = dict(savefig_kwargs)
        png_kwargs.setdefault('dpi', png_dpi)
        ok_png, msg_png = _run_python_snippet(
            _MATPLOTLIB_SAVE_SNIPPET,
            [str(fig_pickle_path), str(png_path), json.dumps(png_kwargs)],
            timeout_seconds,
        )
        if not ok_png:
            print(f'  Warning: PNG export skipped for {png_path.name} ({msg_png})')

        if svg_path is not None:
            svg_kwargs = dict(savefig_kwargs)
            svg_kwargs.setdefault('format', 'svg')
            ok_svg, msg_svg = _run_python_snippet(
                _MATPLOTLIB_SAVE_SNIPPET,
                [str(fig_pickle_path), str(svg_path), json.dumps(svg_kwargs)],
                timeout_seconds,
            )
            if not ok_svg:
                print(f'  Warning: SVG export skipped for {svg_path.name} ({msg_svg})')
