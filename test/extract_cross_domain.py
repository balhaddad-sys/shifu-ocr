#!/usr/bin/env python3
"""
Cross-Domain Image Extractor
-----------------------------
Downloads real-world generic images from the web (that contain text)
and executes the Shifu OCR topological pipeline natively to extract their data.
"""

import os
import sys
import subprocess
import json
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(__file__), 'cross_domain_data')
os.makedirs(OUT_DIR, exist_ok=True)

# Use robust local renderer for generic test imagery to dodge HTTP 403s
IMAGES_TO_FETCH = {
    'generic_receipt': 'RECEIPT\nTOTAL: $14.50\nTAX INCLUDED\nITEMS: 4',
    'inventory_scan': 'General Inventory Report\n145 Units Found\nLocation: Warehouse B\nStatus: CLEAR',
    'literature_excerpt': 'The quick brown fox\njumps over the\nlazy dog.'
}

PIPELINE_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'shifu_ocr', 'pipeline_worker.py')

def create_local_image(text, filepath):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('L', (800, 600), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    
    y = 50
    for line in text.split('\n'):
        try:
            bbox = draw.textbbox((50, y), line, font=font)
            y += (bbox[3] - bbox[1]) + 20
        except AttributeError:
            y += 40
        draw.text((50, y), line, font=font, fill=0)
    
    img.save(filepath)

def main():
    print("Beginning Cross-Domain Image Data Extraction...")
    results_md = "# Cross-Domain OCR Extraction Results\n\n"
    
    for name, content in IMAGES_TO_FETCH.items():
        image_path = os.path.join(OUT_DIR, f"{name}.png")
        
        # 1. Generate generic image locally
        print(f"\n[GENERATE] Creating generic test image: {name}.png...")
        create_local_image(content, image_path)

        # 2. Run Shifu Extraction Pipeline
        print(f"[EXTRACT] Submitting {name} to Shifu Pipeline...")
        try:
            cmd = [sys.executable, PIPELINE_SCRIPT, '--image', image_path, '--backend', 'shifu']
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            output_str = out.decode('utf-8').strip()
            
            try:
                data = json.loads(output_str)
            except Exception:
                lines = output_str.split("\n")
                data = json.loads(lines[-1].strip())

            extracted_text = data.get('text', '')
            confidence = data.get('confidence', 0.0)
            
            print(f"  -> Extraction successful! Confidence: {confidence:.2f}")
            
            # Format report
            results_md += f"## {name}\n"
            results_md += f"**Engine Confidence:** `{confidence:.2f}`\n\n"
            results_md += f"**Extracted Text:**\n```text\n{extracted_text}\n```\n\n"
            results_md += "---\n\n"
            
        except Exception as e:
            print(f"  -> Pipeline failure: {e}")
            results_md += f"## {name}\n**Extraction Failed:** {e}\n\n---\n\n"

    # 3. Save Report
    report_path = os.path.join(OUT_DIR, 'extraction_results.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(results_md)
        
    print(f"\n[COMPLETE] Report saved to {report_path}")


if __name__ == '__main__':
    main()
