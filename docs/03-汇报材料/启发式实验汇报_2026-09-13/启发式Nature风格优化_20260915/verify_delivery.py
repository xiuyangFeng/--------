"""Verify source provenance, final exports, installed resources and deck assembly."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

import pymupdf
from pptx import Presentation
from assemble_report import OUT, STEMS, validate

SKILLS = Path('/public/newhome/cy/.codex/skills')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    records = validate(STEMS)
    sources, alignments, code_audits, output_hashes = {}, [], [], {}
    for meta in records:
        for source in meta['sources']:
            path = source['path']
            expected = source['sha256']
            if path not in sources:
                sources[path] = digest(path)
            assert sources[path] == expected, ('source changed', path)
        path = OUT/'qa'/f'{meta["stem"]}.alignment.json'
        alignment = json.loads(path.read_text())
        assert alignment['verdict'] in ('PASS', 'NOT APPLICABLE'), (path, alignment['verdict'])
        alignments.append({'stem': meta['stem'], 'verdict': alignment['verdict'],
                           'panels': len(alignment.get('layout', {}).get('panels', []))})
        for filename in meta['exports'].values():
            output_hashes[filename] = digest(OUT/filename)

    # The static validator cannot resolve local imports. Feed it the actual
    # shared helper plus actual drawing code, without synthetic directives.
    for script in ('build_modules.py', 'build_quantitative.py', 'build_case_plates.py', 'build_report_figures.py'):
        snapshot = OUT/'qa'/f'{Path(script).stem}_resolved_source.py'
        snapshot.write_text('# Static dependency-expanded audit snapshot; run original build script.\n'
                            + (OUT/'plot_style.py').read_text() + '\n'
                            + (OUT/script).read_text().replace('from __future__ import annotations', ''))
        command = [sys.executable, str(SKILLS/'nature-figure/scripts/validate_figure.py'),
                   str(snapshot), '--backend', 'python', '--json']
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        report = json.loads(run.stdout)
        (OUT/'qa'/f'{Path(script).stem}_source_validation.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        assert run.returncode == 0 and report['summary']['counts']['FAIL'] == 0, (script, report['summary'])
        code_audits.append({'script': script, 'counts': report['summary']['counts'],
                           'warnings': [v for v in report['findings'] if v['level'] == 'WARN']})

    installation = json.loads((OUT/'installation.json').read_text())
    installed = []
    for item in installation['installed_directories']:
        folder = SKILLS/Path(item).name
        assert folder.is_dir() and (folder/'SKILL.md').stat().st_size > 0, folder
        files = [p for p in folder.rglob('*') if p.is_file()]
        assert all(p.stat().st_size > 0 for p in files), ('empty installed file', folder)
        installed.append({'name': folder.name, 'file_count': len(files)})
    assert len(installed) == 20

    decks = []
    for name, stems in [('启发式实验汇报_Nature优化版_20260915', STEMS),
                        ('网络模块_Nature优化版_20260915', ['03_backbone', '04_x5', '05_x11', '06_volume_models'])]:
        pptpath, pdfpath = OUT/f'{name}.pptx', OUT/f'{name}.pdf'
        prs = Presentation(pptpath)
        assert len(prs.slides) == len(stems)
        with pymupdf.open(pdfpath) as pdf:
            assert len(pdf) == len(stems)
            for i, stem in enumerate(stems):
                slide = prs.slides[i]
                assert len(slide.shapes) == 1
                assert hashlib.sha256(slide.shapes[0].image.blob).hexdigest() == digest(OUT/'figures'/f'{stem}.png')
                assert slide.notes_slide.notes_text_frame.text.strip()
                with pymupdf.open(OUT/'figures'/f'{stem}.pdf') as original:
                    assert pdf[i].get_text() == original[0].get_text(), ('merged PDF text', name, i)
        for p in (pptpath, pdfpath):
            output_hashes[p.name] = digest(p)
        decks.append({'name': name, 'pages': len(stems), 'image_bytes_and_pdf_text_match': True})

    for filename in ('plot_style.py', 'build_modules.py', 'build_quantitative.py',
                     'build_case_plates.py', 'build_report_figures.py', 'assemble_report.py', 'verify_delivery.py'):
        output_hashes[filename] = digest(OUT/filename)
    versions = {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable,
                'packages': {package: importlib.metadata.version(package) for package in
                             ('matplotlib', 'numpy', 'pandas', 'scipy', 'vtk', 'Pillow',
                              'python-pptx', 'PyMuPDF', 'lxml', 'XlsxWriter')}}
    (OUT/'qa/runtime_versions.json').write_text(json.dumps(versions, ensure_ascii=False, indent=2)+'\n')
    report = {'status': 'passed', 'figure_count': len(STEMS), 'export_files': len(STEMS)*3,
              'source_hashes_checked': len(sources), 'sources_sha256': sources,
              'output_sha256': output_hashes, 'decks': decks, 'alignment': alignments,
              'static_source_audits': code_audits, 'installed_directories': installed,
              'pdf_collision_failures': 0, 'pdf_collision_warnings': 0,
              'all_pdf_text_audits_passed': True,
              'scope': 'Existing evidence redrawn. No new model inference, training or full-test evaluation.'}
    (OUT/'qa/final_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'figures': len(STEMS), 'exports': len(STEMS)*3,
                      'source_hashes_checked': len(sources), 'decks': decks,
                      'installed_directories': len(installed)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
