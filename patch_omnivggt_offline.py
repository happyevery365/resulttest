"""Allow full-checkpoint inference without redundant DINOv2 initialization downloads.

Keep the original behavior by default. With OMNIVGGT_SKIP_BACKBONE_DOWNLOAD=1,
the caller MUST load a complete OmniVGGT checkpoint using strict=True.
"""
from pathlib import Path
import argparse
import difflib

p=argparse.ArgumentParser()
p.add_argument('repo',type=Path)
a=p.parse_args()
path=a.repo/'omnivggt/models/aggregator.py'
original=path.read_text()
marker='OMNIVGGT_SKIP_BACKBONE_DOWNLOAD'
if marker in original:
    print('Offline initialization patch already present')
else:
    start=original.index('            if patch_embed == "dinov2_vitl14_reg":')
    end=original.index('self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)',
                       original.index('elif patch_embed == "dinov2_vitg2_reg":',start))
    end=original.index('\n',end)+1
    block=original[start:end]
    replacement=('            # Full-checkpoint inference loads all backbone weights afterwards (strict=True).\n'
                 '            import os\n'
                 '            if os.environ.get("OMNIVGGT_SKIP_BACKBONE_DOWNLOAD") != "1":\n'
                 +''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True)))
    updated=original[:start]+replacement+original[end:]
    compile(updated,str(path),'exec')
    diff=''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
                                    fromfile='a/omnivggt/models/aggregator.py',tofile='b/omnivggt/models/aggregator.py'))
    (a.repo/'offline_initialization.patch').write_text(diff)
    path.write_text(updated)
    print(diff)
