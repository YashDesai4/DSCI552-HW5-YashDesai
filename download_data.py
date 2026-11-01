"""Download official UCI data, or copy the four expected files from local ZIPs.

Usage: python download_data.py
       python download_data.py --archives /path/acute.zip /path/communities.zip
Only the expected data/name files are read from ZIPs. Nothing is executed.
"""
from pathlib import Path
import argparse
import io
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'acute': ('https://archive.ics.uci.edu/static/public/184/acute+inflammations.zip',
              ('diagnosis.data', 'diagnosis.names')),
    'communities': ('https://archive.ics.uci.edu/static/public/183/communities+and+crime.zip',
                    ('communities.data', 'communities.names')),
}

def copy_expected(archive, root=ROOT):
    copied=[]
    with zipfile.ZipFile(archive) as z:
        for folder, (_, expected) in SOURCES.items():
            for basename in expected:
                matches=[name for name in z.namelist() if Path(name).name==basename and not name.endswith('/')]
                if len(matches)>1:
                    raise ValueError(f'Multiple copies of {basename} in ZIP; use the original UCI archive.')
                if matches:
                    target=root/'data'/folder/basename
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(matches[0]))
                    copied.append(str(target.relative_to(root)))
    return copied

def prepare_data(archives=(), allow_download=True, root=ROOT):
    root=Path(root)
    candidates=[Path(x) for x in archives]
    for folder in [root,root/'data',root/'upload']:
        if folder.exists(): candidates.extend(folder.glob('*.zip'))
    for path in dict.fromkeys(candidates):
        if path.is_file():
            copied=copy_expected(path,root)
            if copied: print('Read local ZIP:',path.name,'->',copied)
    for folder,(url,expected) in SOURCES.items():
        if all((root/'data'/folder/name).is_file() for name in expected): continue
        if allow_download:
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'DSCI552-assignment-data-loader/1.0'})
                with urllib.request.urlopen(request,timeout=45) as response: payload=response.read()
                copy_expected(io.BytesIO(payload),root)
            except Exception as exc:
                print(f'Download unavailable for {folder}: {type(exc).__name__}: {exc}')
    missing=[str(Path('data')/folder/name) for folder,(_,expected) in SOURCES.items()
             for name in expected if not (root/'data'/folder/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing assignment data: '+', '.join(missing)+
          '. Download the two original UCI ZIPs, put them beside this notebook, and rerun. '
          'Alternatively run: python download_data.py --archives PATH_TO_ACUTE_ZIP PATH_TO_COMMUNITIES_ZIP')
    return {folder:{name:root/'data'/folder/name for name in expected}
            for folder,(_,expected) in SOURCES.items()}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',nargs='*',default=[])
    args=parser.parse_args()
    prepare_data(args.archives)
    print('Both datasets are ready under data/.')
