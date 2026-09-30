"""Run the existing PDF collision auditor with Python 3.8's installed PyMuPDF.

Only three runtime typing aliases are adapted to typing.Tuple. The auditing
logic and thresholds are unchanged; no source in the skill is modified.
"""
from pathlib import Path
p=Path('/public/newhome/cy/.codex/skills/nature-figure/scripts/audit_figure_collisions.py')
s=p.read_text()
for name in ('Rect','Point','Segment'):
 s=s.replace('\n'+name+' = tuple[','\n'+name+" = __import__('typing').Tuple[")
exec(compile(s,str(p),'exec'),globals())
