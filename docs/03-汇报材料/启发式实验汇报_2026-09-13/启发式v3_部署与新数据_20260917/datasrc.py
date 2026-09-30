"""Read-only loaders for the v3 report figures.

Every number drawn in this package is read from an experiment artefact under
``training_wss_min/experiments`` or ``docs/02-推进与变更/_archive/WSS_PINN``.  No figure
script hard-codes an experiment number; ``dump()`` writes what was actually
used into ``source_data/<stem>.json`` so a reader can re-check a value without
re-running the analysis.
"""
from pathlib import Path
import csv, json, re

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'training_wss_min').is_dir())
EXP = ROOT / 'training_wss_min' / 'experiments'
RCR = ROOT / 'docs' / '02-推进与变更' / 'WSS_PINN' / 'RCR出口面积录入核查_2026-09-15'

# ---------------------------------------------------------------- experiments
W6 = EXP / 'wss_local_wave6_20260915'
W6B = EXP / 'wss_local_wave6b_20260915'
FOLLOW = EXP / 'wss_x5d_inference_followup_20260915'
W1 = EXP / 'wss_v51_wave1_20260916'
W1R = EXP / 'wss_v51_wave1r_20260916'
W2A = EXP / 'wss_v51_wave2a_20260916'
W2B = EXP / 'wss_v51_wave2b_20260916'
LONG = EXP / 'wss_x5d_longitudinal_20260917'
WAVE1 = EXP / 'wss_local_wave1_20260912'
WAVE1B = EXP / 'wss_local_wave1b_20260912'
WAVE5 = EXP / 'wss_local_wave5_20260913'

_used: list[Path] = []


def load(path):
    """Read a JSON artefact and remember it as a source of the current figure."""
    path = Path(path)
    _used.append(path)
    return json.loads(path.read_text())


def text(path):
    path = Path(path)
    _used.append(path)
    return path.read_text()


def sources():
    """Paths touched since the last dump(), in first-use order."""
    seen, out = set(), []
    for p in _used:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def dump(stem, payload):
    """Write the extracted values + their provenance next to the figure."""
    body = {'stem': stem,
            'sources': [str(p.relative_to(ROOT)) for p in sources()],
            'values': payload}
    (OUT / 'source_data' / f'{stem}.json').write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + '\n')
    out = list(sources())
    _used.clear()
    return out


# ----------------------------------------------------------- deployment scans
def _ens(row, pa_mean=False):
    key = 'ensemble_pa_mean' if pa_mean and 'ensemble_pa_mean' in row else 'ensemble'
    ref = 'reference_' + key
    return row[key], row[ref]


def stl_rows(path, pa_mean=False):
    """[(target_mm, actual_mm, points, ensemble R2, same-point reference R2)]."""
    d = load(path)
    out = []
    for r in d['rows']:
        ens, ref = _ens(r, pa_mean)
        out.append({'target': r['spacing_mm'],
                    'actual': r['diag_median']['spacing_mm'],
                    'points': r['points_median'],
                    'r2': ens['pa_r2_cb'], 'ref': ref['pa_r2_cb'],
                    'high': ens['high_wss_r2'], 'iou': ens['top10_iou_mean'],
                    'norm': ens['norm_r2_cb']})
    return sorted(out, key=lambda x: x['actual'])


def resample_rows(path, pa_mean=False):
    """[(kept fraction, variant, ensemble R2, same-point reference R2)]."""
    d = load(path)
    out = []
    for r in d['rows']:
        ens, ref = _ens(r, pa_mean)
        out.append({'fraction': r['fraction'], 'variant': r['variant'],
                    'points': r['points_median'],
                    'r2': ens['pa_r2_cb'], 'ref': ref['pa_r2_cb']})
    return out


def geometry_rows(path, pa_mean=False):
    """[(variant, vertex displacement, ensemble R2, same-point reference R2)]."""
    d = load(path)
    out = []
    for r in d['rows']:
        ens, ref = _ens(r, pa_mean)
        diag = r['diag_median']
        out.append({'variant': r['variant'],
                    'shift': diag.get('vertex_disp_mm_median'),
                    'shift_p95': diag.get('vertex_disp_mm_p95'),
                    'r2': ens['pa_r2_cb'], 'ref': ref['pa_r2_cb'],
                    'high': ens['high_wss_r2']})
    return out


def decomposition_rows(path, pa_mean=False):
    d = load(path)
    out = []
    for r in d['rows']:
        ens, ref = _ens(r, pa_mean)
        out.append({'fraction': r['fraction'], 'variant': r['variant'],
                    'support': r.get('support'), 'patch': r.get('patch'),
                    'r2': ens['pa_r2_cb'], 'ref': ref['pa_r2_cb']})
    return out


def tta_rows(path):
    d = load(path)
    rows = [{'label': r['label'],
             'r2': r.get('ensemble_pa_mean', r['ensemble'])['pa_r2_cb']}
            for r in d['rows']]
    return rows, d


def wall_spacing():
    """Per-case CFD wall node spacing, grouped by cohort."""
    rows = load(W6 / 'offline' / 'cfd_wall_spacing.json')
    return rows


# --------------------------------------------------------------- RCR audit
def rcr_audit():
    """Per-outlet RCR entry audit.  The file is CRLF; strip \\r before use."""
    path = RCR / 'rcr_area_audit.csv'
    _used.append(path)
    raw = path.read_text().replace('\r\n', '\n').replace('\r', '\n')
    return list(csv.DictReader(raw.splitlines()))


def rcr_corrected():
    path = RCR / 'rcr_corrected_udf_values.csv'
    _used.append(path)
    raw = path.read_text().replace('\r\n', '\n').replace('\r', '\n')
    return list(csv.DictReader(raw.splitlines()))


# ------------------------------------------------------------ text artefacts
def kv_from_text(path, pattern, cast=float):
    """Pull ``pattern`` (one capture group) out of a plain-text analysis dump."""
    body = text(path)
    m = re.search(pattern, body)
    if not m:
        raise KeyError(f'{pattern!r} not found in {path}')
    return cast(m.group(1))
