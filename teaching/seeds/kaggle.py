"""
Shifu Kaggle Data Seeder
Fetches, processes, and exports datasets from Kaggle for cross-domain teaching.

Usage:
    python -m teaching.seeds.kaggle --domain medical --output seeds_out/
    python -m teaching.seeds.kaggle --all --output seeds_out/
    python -m teaching.seeds.kaggle --list

Requirements:
    pip install kaggle pandas
    Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place kaggle.json in ~/.kaggle/
"""

import os
import sys
import json
import argparse
import hashlib
import re
from pathlib import Path

# ─── Domain → Kaggle Dataset Mapping ────────────────────────────────────

KAGGLE_DATASETS = {
    "medical": [
        {
            "dataset": "tboyle10/medicaltranscriptions",
            "description": "Medical transcription samples across specialties",
            "text_columns": ["transcription", "description"],
            "max_samples": 5000,
        },
        {
            "dataset": "chaitanyakck/medical-text",
            "description": "Medical text classification dataset",
            "text_columns": ["Text"],
            "max_samples": 5000,
        },
        {
            "dataset": "finalepoch/medical-dialog",
            "description": "Medical dialogue dataset for clinical conversation",
            "text_columns": ["utterances"],
            "max_samples": 3000,
        },
        {
            "dataset": "itachi9604/disease-symptom-description-dataset",
            "description": "Disease symptom descriptions",
            "text_columns": ["Description"],
            "max_samples": 2000,
        },
    ],
    "legal": [
        {
            "dataset": "amohankumar/legal-text-classification-dataset",
            "description": "Legal document classification",
            "text_columns": ["case_text"],
            "max_samples": 5000,
        },
        {
            "dataset": "shivamb/legal-case-document-summarization",
            "description": "Legal case documents with summaries",
            "text_columns": ["original_text", "summary"],
            "max_samples": 3000,
        },
    ],
    "financial": [
        {
            "dataset": "ankurzing/sentiment-analysis-for-financial-news",
            "description": "Financial news sentiment analysis",
            "text_columns": ["Sentence"],
            "max_samples": 5000,
        },
        {
            "dataset": "jeet2016/us-financial-news-articles",
            "description": "US financial news articles",
            "text_columns": ["content", "title"],
            "max_samples": 5000,
        },
        {
            "dataset": "vivekrathi055/sentiment-analysis-on-financial-tweets",
            "description": "Financial tweets for sentiment analysis",
            "text_columns": ["tweet"],
            "max_samples": 3000,
        },
    ],
    "scientific": [
        {
            "dataset": "Cornell-University/arxiv",
            "description": "ArXiv paper metadata and abstracts",
            "text_columns": ["abstract", "title"],
            "max_samples": 5000,
        },
        {
            "dataset": "benhamner/nips-papers",
            "description": "NIPS/NeurIPS conference papers",
            "text_columns": ["paper_text", "title"],
            "max_samples": 3000,
        },
    ],
    "engineering": [
        {
            "dataset": "stackoverflow/stacksample",
            "description": "Stack Overflow questions and answers",
            "text_columns": ["Body", "Title"],
            "max_samples": 5000,
        },
    ],
    "education": [
        {
            "dataset": "CoderSaty/education-dataset",
            "description": "Education sector data",
            "text_columns": ["text"],
            "max_samples": 3000,
        },
    ],
    "retail": [
        {
            "dataset": "olistbr/brazilian-ecommerce",
            "description": "Brazilian e-commerce public dataset",
            "text_columns": ["review_comment_message"],
            "max_samples": 5000,
        },
        {
            "dataset": "carrie1/ecommerce-data",
            "description": "E-commerce transaction data",
            "text_columns": ["Description"],
            "max_samples": 5000,
        },
    ],
    "logistics": [
        {
            "dataset": "laurinbrechter/supply-chain-data",
            "description": "Supply chain shipment data",
            "text_columns": ["Product Description"],
            "max_samples": 3000,
        },
    ],
    "general": [
        {
            "dataset": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "description": "IMDB movie reviews for general language",
            "text_columns": ["review"],
            "max_samples": 5000,
        },
        {
            "dataset": "therohk/million-headlines",
            "description": "ABC news headlines",
            "text_columns": ["headline_text"],
            "max_samples": 5000,
        },
        {
            "dataset": "abisheksudarshan/topic-modeling-for-research-articles",
            "description": "Research article abstracts",
            "text_columns": ["ABSTRACT"],
            "max_samples": 3000,
        },
    ],
    "handwriting": [
        {
            "dataset": "landlord/handwriting-recognition",
            "description": "Handwriting recognition dataset",
            "text_columns": ["IDENTITY"],
            "max_samples": 3000,
        },
        {
            "dataset": "sachinpatel21/az-handwritten-alphabets-in-csv-format",
            "description": "Handwritten alphabets in CSV",
            "text_columns": ["label"],
            "max_samples": 2000,
        },
    ],
}

# ─── Text Processing ────────────────────────────────────────────────────

def clean_text(text):
    """Clean and normalize text for teaching."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove very long sequences of special characters
    text = re.sub(r'[^\w\s.,;:!?\'"()-]{5,}', '', text)
    return text


def extract_sentences(text, min_length=10, max_length=500):
    """Split text into sentences suitable for teaching."""
    if not text:
        return []
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if min_length <= len(s) <= max_length:
            sentences.append(s)
    return sentences


def generate_ocr_confusion_pairs(text, confusion_map=None):
    """Generate synthetic OCR confusion pairs for teaching calibration."""
    if confusion_map is None:
        confusion_map = {
            'O': '0', '0': 'O', 'l': '1', '1': 'l', 'I': 'l',
            'S': '5', '5': 'S', 'B': '8', '8': 'B', 'G': '6',
            'Z': '2', 'n': 'r', 'm': 'rn', 'e': 'c', 'h': 'b',
            'D': 'O', 't': 'f', 'a': 'o', 'd': 'cl',
        }
    pairs = []
    words = text.split()
    for word in words:
        if len(word) < 3:
            continue
        # Apply 1-2 confusions per word
        corrupted = list(word)
        applied = 0
        for i, char in enumerate(corrupted):
            if char in confusion_map and applied < 2:
                # 30% chance of applying each confusion
                h = hashlib.md5(f"{word}{i}".encode()).hexdigest()
                if int(h[:2], 16) < 77:  # ~30%
                    corrupted[i] = confusion_map[char]
                    applied += 1
        corrupted_word = ''.join(corrupted)
        if corrupted_word != word:
            pairs.append({
                "input": corrupted_word,
                "expected": word,
                "confidence": 0.3 + 0.4 * (1 - applied / max(len(word), 1)),
            })
    return pairs


# ─── Kaggle Fetcher ─────────────────────────────────────────────────────

class KaggleFetcher:
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', '.cache', 'kaggle'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_api(self):
        """Get authenticated Kaggle API client."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return api
        except ImportError:
            print("ERROR: kaggle package not installed. Run: pip install kaggle")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Kaggle authentication failed: {e}")
            print("Ensure KAGGLE_USERNAME and KAGGLE_KEY are set, or ~/.kaggle/kaggle.json exists")
            sys.exit(1)

    def fetch_dataset(self, dataset_ref, force=False):
        """Download a Kaggle dataset to cache."""
        cache_path = self.cache_dir / dataset_ref.replace('/', '_')
        if cache_path.exists() and not force:
            print(f"  Using cached: {dataset_ref}")
            return cache_path

        print(f"  Downloading: {dataset_ref}")
        api = self._get_api()
        api.dataset_download_files(dataset_ref, path=str(cache_path), unzip=True)
        return cache_path

    def load_csv(self, dataset_path, text_columns, max_samples=5000):
        """Load text data from CSV files in a dataset directory."""
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: pandas not installed. Run: pip install pandas")
            return []

        texts = []
        csv_files = list(Path(dataset_path).rglob("*.csv"))

        for csv_file in csv_files[:5]:  # Limit to 5 CSV files per dataset
            try:
                df = pd.read_csv(csv_file, nrows=max_samples, encoding='utf-8',
                                 on_bad_lines='skip')
            except Exception:
                try:
                    df = pd.read_csv(csv_file, nrows=max_samples, encoding='latin-1',
                                     on_bad_lines='skip')
                except Exception as e:
                    print(f"  Warning: Could not read {csv_file.name}: {e}")
                    continue

            for col in text_columns:
                if col in df.columns:
                    col_texts = df[col].dropna().astype(str).tolist()
                    texts.extend(col_texts[:max_samples])

            if len(texts) >= max_samples:
                break

        return texts[:max_samples]

    def load_json(self, dataset_path, text_columns, max_samples=5000):
        """Load text data from JSON files in a dataset directory."""
        texts = []
        json_files = list(Path(dataset_path).rglob("*.json"))

        for json_file in json_files[:5]:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    # Try line-delimited JSON first
                    for line_num, line in enumerate(f):
                        if line_num >= max_samples:
                            break
                        try:
                            obj = json.loads(line.strip())
                            for col in text_columns:
                                if col in obj and isinstance(obj[col], str):
                                    texts.append(obj[col])
                        except json.JSONDecodeError:
                            break
            except Exception as e:
                print(f"  Warning: Could not read {json_file.name}: {e}")

        return texts[:max_samples]

    def process_domain(self, domain, output_dir, force=False):
        """Fetch and process all datasets for a domain."""
        if domain not in KAGGLE_DATASETS:
            print(f"Unknown domain: {domain}")
            return None

        datasets = KAGGLE_DATASETS[domain]
        all_sentences = []
        all_pairs = []

        for ds_config in datasets:
            dataset_ref = ds_config["dataset"]
            print(f"\nProcessing: {dataset_ref}")

            try:
                dataset_path = self.fetch_dataset(dataset_ref, force=force)
            except Exception as e:
                print(f"  Error fetching {dataset_ref}: {e}")
                continue

            # Load texts
            text_cols = ds_config["text_columns"]
            max_samples = ds_config.get("max_samples", 5000)

            texts = self.load_csv(dataset_path, text_cols, max_samples)
            if not texts:
                texts = self.load_json(dataset_path, text_cols, max_samples)

            if not texts:
                print(f"  No text data found in {dataset_ref}")
                continue

            # Process texts
            for text in texts:
                cleaned = clean_text(text)
                sentences = extract_sentences(cleaned)
                all_sentences.extend(sentences)

                # Generate confusion pairs for calibration
                pairs = generate_ocr_confusion_pairs(cleaned)
                all_pairs.extend(pairs)

            print(f"  Extracted {len(all_sentences)} sentences, {len(all_pairs)} confusion pairs")

        # Deduplicate
        all_sentences = list(dict.fromkeys(all_sentences))

        # Save output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"kaggle_{domain}_seed.json"
        result = {
            "source": "kaggle",
            "domain": domain,
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "stats": {
                "total_sentences": len(all_sentences),
                "total_pairs": len(all_pairs),
                "datasets_processed": len(datasets),
            },
            "sentences": all_sentences[:10000],  # Cap at 10K
            "pairs": all_pairs[:5000],            # Cap at 5K
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {output_file}")
        print(f"  Sentences: {len(result['sentences'])}")
        print(f"  Pairs: {len(result['pairs'])}")

        return result


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shifu Kaggle Data Seeder")
    parser.add_argument("--domain", type=str, help="Domain to fetch (e.g., medical, legal)")
    parser.add_argument("--all", action="store_true", help="Fetch all domains")
    parser.add_argument("--list", action="store_true", help="List available domains and datasets")
    parser.add_argument("--output", type=str, default="seeds_out",
                       help="Output directory (default: seeds_out)")
    parser.add_argument("--force", action="store_true", help="Force re-download cached datasets")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for downloads")
    args = parser.parse_args()

    if args.list:
        print("\n=== Available Kaggle Datasets by Domain ===\n")
        for domain, datasets in sorted(KAGGLE_DATASETS.items()):
            print(f"  {domain}:")
            for ds in datasets:
                print(f"    - {ds['dataset']}: {ds['description']}")
            print()
        print(f"Total: {sum(len(v) for v in KAGGLE_DATASETS.values())} datasets across "
              f"{len(KAGGLE_DATASETS)} domains")
        return

    fetcher = KaggleFetcher(cache_dir=args.cache_dir)

    if args.all:
        for domain in KAGGLE_DATASETS:
            print(f"\n{'='*60}")
            print(f"  Domain: {domain}")
            print(f"{'='*60}")
            fetcher.process_domain(domain, args.output, force=args.force)
    elif args.domain:
        fetcher.process_domain(args.domain, args.output, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
