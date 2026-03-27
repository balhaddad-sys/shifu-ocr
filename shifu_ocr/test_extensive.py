"""Extensive OCR test across all system fonts."""
import os, sys, glob, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))
from shifu_ocr.engine import ShifuOCR

ocr = ShifuOCR.load(os.path.join(os.path.dirname(__file__), 'trained_model.json'))

font_dirs = ['C:/Windows/Fonts', os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts')]
all_fonts = []
for d in font_dirs:
    if os.path.exists(d):
        all_fonts.extend(glob.glob(os.path.join(d, '*.ttf')))
        all_fonts.extend(glob.glob(os.path.join(d, '*.TTF')))
seen = set()
usable = []
for f in sorted(all_fonts):
    base = os.path.basename(f).lower()
    if base in seen: continue
    seen.add(base)
    try:
        font = ImageFont.truetype(f, 36)
        img = Image.new('L', (100,50), 255)
        draw = ImageDraw.Draw(img)
        draw.text((5,5), 'Abc', fill=0, font=font)
        if (np.array(img) < 128).sum() > 20:
            usable.append(f)
    except Exception: pass
print(f'Testing on {len(usable)} fonts', flush=True)

tests = [
    'stroke patient admitted',
    'Levetiracetam 500mg daily',
    'Dr Hisham Saleh',
    'bilateral weakness noted',
    'chest infection pneumonia',
    'seizure epilepsy status',
    'potassium 4.5 sodium 139',
    'aspirin clopidogrel warfarin',
    'Abdullah Mohammed Hassan',
    'pneumonia respiratory failure',
    'metoprolol bisoprolol atenolol',
    'ischemic hemorrhagic stroke',
    'creatinine 89 urea 5.2',
    'discharge home stable',
    'insulin glargine 20 units',
    'blood pressure heart rate',
    'atrial fibrillation rhythm',
    'Noura Bazzah ward 3',
    'paracetamol ibuprofen codeine',
    'meningitis encephalitis',
    'hemiparesis hemiplegia',
    'ventilator oxygen bipap',
    'glucose 6.5 hba1c 7.2',
    'omeprazole pantoprazole',
    'acute chronic subacute',
]

total_cc = 0; total_ct = 0; total_wc = 0; total_wt = 0
font_scores = {}

for fi, fp in enumerate(usable):
    fn = os.path.basename(fp)
    try: font = ImageFont.truetype(fp, 36)
    except Exception: continue
    fc = 0; ft = 0; fwc = 0; fwt = 0
    for text in tests:
        img = Image.new('L', (700, 60), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, fill=0, font=font)
        result = ocr.read_line(np.array(img))
        pred = result['text'].lower().replace(' ','')
        truth = text.lower().replace(' ','')
        m = sum(1 for a,b in zip(pred, truth) if a == b)
        t = max(len(pred), len(truth))
        fc += m; ft += t; total_cc += m; total_ct += t
        for pw, tw in zip(result['text'].lower().split(), text.lower().split()):
            fwt += 1; total_wt += 1
            if pw == tw: fwc += 1; total_wc += 1
    font_scores[fn] = (100*fc/max(ft,1), 100*fwc/max(fwt,1))
    if (fi+1) % 20 == 0:
        print(f'  [{fi+1}/{len(usable)}] char: {100*total_cc/max(total_ct,1):.1f}% word: {100*total_wc/max(total_wt,1):.1f}%', flush=True)

print(f'\n========================================', flush=True)
print(f'  EXTENSIVE OCR TEST RESULTS', flush=True)
print(f'========================================', flush=True)
print(f'  Fonts tested:     {len(font_scores)}', flush=True)
print(f'  Test sentences:   {len(tests)}', flush=True)
print(f'  Total characters: {total_ct}', flush=True)
print(f'  Total words:      {total_wt}', flush=True)
print(f'  CHARACTER ACCURACY: {100*total_cc/total_ct:.1f}%', flush=True)
print(f'  WORD ACCURACY:      {100*total_wc/total_wt:.1f}%', flush=True)
print(f'========================================', flush=True)
sf = sorted(font_scores.items(), key=lambda x: x[1][0], reverse=True)
print(f'\nTop 10 fonts:', flush=True)
for n,(ca,wa) in sf[:10]: print(f'  {n:30s} char:{ca:5.0f}%  word:{wa:5.0f}%', flush=True)
print(f'\nBottom 10 fonts:', flush=True)
for n,(ca,wa) in sf[-10:]: print(f'  {n:30s} char:{ca:5.0f}%  word:{wa:5.0f}%', flush=True)
