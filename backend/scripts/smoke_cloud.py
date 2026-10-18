"""Optional REAL cloud smoke test; sends a synthetic diagram, never camera data.

Run from ECHOSIGHT: ../.venv/bin/python -m backend.scripts.smoke_cloud --live
"""
import argparse
import asyncio
import io
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from backend.services.intelligence import Intelligence
from backend.sessions import Observation


async def smoke():
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    image = Image.new('RGB', (640, 480), 'white')
    ImageDraw.Draw(image).rectangle((180, 130, 460, 350), fill='blue')
    output = io.BytesIO()
    image.save(output, 'JPEG')
    observation = Observation('synthetic-smoke', output.getvalue(), [],
                              {'action': 'CAUTION', 'message': 'Synthetic test diagram; no physical navigation evidence.'},
                              'Synthetic blue rectangle on white.', time.monotonic(), 'upload', False)
    client = Intelligence.from_environment()
    try:
        result = await client.enrich('This is a synthetic test diagram. Describe the visible shape and its color.', observation, upload=True)
        # Only application-owned telemetry and synthetic-image output; no headers or IDs.
        report = {key: result.get(key) for key in ('sources', 'cloud', 'elapsed_ms', 'answer')}
        print(json.dumps(report, indent=2))
        return result['cloud'].get('amd') == 'success' and result['cloud'].get('evolus') == 'success'
    finally:
        await client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Send one synthetic image to AMD and its observations to Evolus.')
    if not parser.parse_args().live:
        parser.error('Pass --live to make real provider requests.')
    raise SystemExit(0 if asyncio.run(smoke()) else 1)
