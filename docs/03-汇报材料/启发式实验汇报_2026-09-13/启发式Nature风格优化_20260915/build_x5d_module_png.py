"""Regenerate only the X5D/X11 module PNGs; never write PDF or SVG."""
from pathlib import Path
import hashlib, json, sys
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path('/public/newhome/cy/.codex/skills/nature-figure/scripts')))
from audit_panel_alignment import require_matplotlib_panel_alignment
import build_modules as modules


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_png_only(fig, stem, *, claim='', sources=(), axes=None, row_groups=None,
                  column_groups=None, exemptions=(), note=''):
    """Preserve the existing figure geometry while exporting PNG only."""
    stem = {'04_x5': '04_x5', '05_x11': '05_x11'}[stem]
    figure_path = OUT / 'figures' / f'{stem}.png'
    qa_path = OUT / 'qa' / f'{stem}_x5d_png'
    figure_path.parent.mkdir(exist_ok=True)
    qa_path.parent.mkdir(exist_ok=True)
    opts = {'axes': axes, 'exemptions': exemptions}
    if row_groups is not None:
        opts['row_groups'] = row_groups
    if column_groups is not None:
        opts['column_groups'] = column_groups
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=str(qa_path) + '.alignment.json', tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True, **opts)
    assert alignment.get('verdict') in ('PASS', 'NOT APPLICABLE')
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    metadata = {
        'stem': stem, 'claim': claim, 'export': str(figure_path.relative_to(OUT)),
        'format': 'PNG only', 'resolution_px': [4800, 2700],
        'sources': [{'path': str(Path(source).resolve()), 'sha256': digest(source)} for source in sources],
        'alignment': alignment, 'note': note,
    }
    (qa_path.with_suffix('.metadata.json')).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(stem, 'PNG only', figure_path, alignment.get('verdict'), flush=True)
    return metadata


modules.save = save_png_only
modules.diagram_note = lambda *args, **kwargs: None
modules.x5()
modules.x11()
