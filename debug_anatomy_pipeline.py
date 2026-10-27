from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from model_service import model_service, strip_internal_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dental anatomy pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to a panoramic X-ray image.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise SystemExit(f"Image does not exist: {image_path}")

    model_service.load()
    if not model_service.ready:
        raise SystemExit(model_service.load_error or "Models are not ready.")

    analysis = model_service.analyze(image_path)
    teeth = analysis["teeth"]
    summary = analysis["anatomy_summary"]
    removed_duplicates = analysis["removed_duplicates"]
    suspicious_teeth = [tooth for tooth in teeth if tooth.get("anatomy_status") == "suspicious"]

    print(f"raw tooth count: {summary['total_teeth_detected_raw']}")
    print(f"filtered tooth count: {summary['total_teeth_after_duplicate_filter']}")
    print(f"removed duplicates: {len(removed_duplicates)}")
    if removed_duplicates:
        print(json.dumps(removed_duplicates, indent=2))
    print(f"upper count: {summary['upper_count']}")
    print(f"lower count: {summary['lower_count']}")
    print(f"suspicious teeth: {[tooth.get('tooth_number') for tooth in suspicious_teeth]}")
    print(f"upper sequence: {summary['sequence']['upper']['level']} - {summary['sequence']['upper']['detail']}")
    print(f"lower sequence: {summary['sequence']['lower']['level']} - {summary['sequence']['lower']['detail']}")

    preview = {
        "teeth": strip_internal_masks(teeth[:5]),
        "removed_duplicates": removed_duplicates,
        "anatomy_summary": summary,
        "findings_preview": strip_internal_masks(analysis["findings"][:5]),
    }
    print("final JSON preview:")
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
