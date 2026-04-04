#!/usr/bin/env python3
"""
Cross-Domain Data Harvester
----------------------------
Downloads raw generic text and literature (Gutenberg/Wiki excerpts)
and constructs OCR test images to validate Shifu's performance
against non-medical biases.
"""

import os
import urllib.request
import textwrap
import random
from PIL import Image, ImageDraw, ImageFont

# Ensure the output directory exists
OUT_DIR = os.path.join(os.path.dirname(__file__), 'cross_domain_data')
os.makedirs(OUT_DIR, exist_ok=True)

# Sources
SOURCES = {
    'gutenberg_alice': 'https://www.gutenberg.org/files/11/11-0.txt',
    'gutenberg_frankenstein': 'https://www.gutenberg.org/files/84/84-0.txt',
    'gutenberg_pride': 'https://www.gutenberg.org/files/1342/1342-0.txt'
}

def clean_text(raw_text):
    """Basic structural cleaning for raw text."""
    # Split text to look for start boundary in Gutenberg texts
    lines = raw_text.split('\r\n')
    start = 0
    end = len(lines)
    for i, line in enumerate(lines):
        if 'START OF THE PROJECT GUTENBERG' in line or 'START OF THIS PROJECT GUTENBERG' in line:
            start = i + 1
        if 'END OF THE PROJECT GUTENBERG' in line or 'END OF THIS PROJECT GUTENBERG' in line:
            end = i
            break
            
    content = '\n'.join(lines[start:end])
    # Remove excessive multiline breaks
    import re
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content

def create_sample_image(text_chunk, filename):
    """
    Renders text onto a blank white image resembling a scanned document
    so it can be ingested by the OCR topological pipeline.
    """
    WIDTH, HEIGHT = 1800, 2400
    img = Image.new('L', (WIDTH, HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    
    # Attempt to load a default font
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
        
    y_text = 100
    margin = 100
    
    # Process text chunk into lines
    lines = text_chunk.split('\n')
    for line in lines:
        if line.strip() == '':
            y_text += 60  # paragraph spacing
            continue
            
        # Wrap the line safely
        wrapped = textwrap.wrap(line, width=70)
        for w in wrapped:
            # We use textbbox if available, otherwise fallback bounds estimating
            try:
                bbox = draw.textbbox((margin, y_text), w, font=font)
                y_text += (bbox[3] - bbox[1]) + 20
            except AttributeError:
                y_text += 60
                
            draw.text((margin, y_text), w, font=font, fill=0)
            
            # Stop if we hit image bottom
            if y_text >= HEIGHT - 100:
                break
        if y_text >= HEIGHT - 100:
            break
            
    img.save(os.path.join(OUT_DIR, filename))
    print(f"Generated OCR test image: {filename}")


print("Starting cross-domain literature download...")
for name, url in SOURCES.items():
    try:
        print(f"Fetching {name} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        raw_text = response.read().decode('utf-8')
        
        cleaned = clean_text(raw_text)
        txt_path = os.path.join(OUT_DIR, f"{name}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"  Saved raw text corpus: {txt_path}")
        
        # Grab a random meaningful chunk of text from the middle of the book
        # to generate an OCR image
        words = cleaned.split()
        if len(words) > 1000:
            seed_start = random.randint(len(words)//4, len(words)//2)
            chunk = ' '.join(words[seed_start:seed_start+500])
            create_sample_image(chunk, f'ocr_test_{name}.png')
            
    except Exception as e:
        print(f"Error fetching {name}: {e}")

print("Cross-domain seed harvesting complete!")
