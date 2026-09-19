#!/usr/bin/env python3
"""Render a local VSDX through Sauron's disposable worker, without indexing it."""
import argparse
import asyncio
import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def inspect(source, destination):
    from src.config import settings
    from src.figures import storage
    from src.ingestion.isolation import extract_in_worker
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Choose an empty output directory to keep inspections separate')
    with tempfile.TemporaryDirectory(prefix='sauron-visio-inspect-') as temporary:
        settings.extraction_work_dir = str(Path(temporary) / 'worker')
        storage.FIGURE_ROOT = Path(temporary) / 'assets'
        result = await extract_in_worker(source.resolve(), source.name)
        (destination / 'source-text.txt').write_text(result.parsed.text, encoding='utf-8')
        records = []
        for figure in result.office.figures if result.office else []:
            record = dataclasses.asdict(figure)
            for variant, asset in record['assets'].items():
                original = Path(result.figure_staging) / asset['key']
                storage.FigureStore.validate_file(original, asset['sha256'], asset['bytes'])
                filename = f"page-{figure.page + 1}-{variant}.png"
                shutil.copyfile(original, destination / filename)
                asset['file'] = filename
            records.append(record)
        report = {'source': source.name, 'warnings': result.warnings, 'figures': records,
                  'renderer': 'libvisio + librsvg', 'fidelity': 'Converted preview; inspect warnings and visual layout.'}
        (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'output': str(destination.resolve()), 'pages_rendered': len(records), 'warnings': result.warnings}, indent=2))
        return 0 if records else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.source.suffix.lower() != '.vsdx':
        parser.error('Only .vsdx is supported')
    raise SystemExit(asyncio.run(inspect(arguments.source, arguments.output)))
