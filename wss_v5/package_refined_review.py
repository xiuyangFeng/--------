"""Package the current offline review desk, including case-page return links."""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import posixpath
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile, ZIP_DEFLATED


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k,v in attrs if k in {'href','src'} and v)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();review=args.review.resolve();base=review.parent
    summary=json.loads((review/'summary.json').read_text())
    files={p.relative_to(base).as_posix():p.read_bytes() for p in review.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    for row in summary['manual_cases']:
        rel='cases/'+row['canonical_id'].replace('/','__')+'.html'
        files[rel]=(base/rel).read_bytes()
    rel='assets/plotly-2.35.2.min.js';files[rel]=(base/rel).read_bytes()
    target=review.name+'/index.html'
    files['index.html']=(f'<!doctype html><meta charset="utf-8"><title>几何审查入口</title>'
                         f'<meta http-equiv="refresh" content="0;url={target}">'
                         f'<a href="{target}">打开当前几何审查与人工待办</a>').encode()
    missing=[]
    for name,data in files.items():
        if not name.endswith('.html'):continue
        parser=Links();parser.feed(data.decode('utf-8'))
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path:continue
            resolved=posixpath.normpath(posixpath.join(posixpath.dirname(name),unquote(u.path)))
            if resolved not in files:missing.append([name,link,resolved])
    if missing:raise RuntimeError(f'Offline links missing: {missing}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.zip.tmp')
    with ZipFile(tmp,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(files.items()):archive.writestr(name,data)
    with ZipFile(tmp) as archive:
        assert archive.testzip() is None
        assert all(archive.read(name)==data for name,data in files.items())
    tmp.replace(args.output)
    result={'source':str(review),'output':str(args.output.resolve()),'files':len(files),
            'bytes':args.output.stat().st_size,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'crc_passed':True,'all_static_html_links_resolve':True,'contents_match_current_files':True}
    args.output.with_suffix('.validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
