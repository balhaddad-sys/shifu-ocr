"""
Shifu HuggingFace Data Seeder
Fetches, processes, and exports datasets from HuggingFace Hub for cross-domain teaching.

Usage:
    python -m teaching.seeds.huggingface --domain medical --output seeds_out/
    python -m teaching.seeds.huggingface --all --output seeds_out/
    python -m teaching.seeds.huggingface --list

Requirements:
    pip install datasets huggingface_hub
"""

import os
import sys
import json
import argparse
import re
import hashlib
from pathlib import Path

# ─── Domain → HuggingFace Dataset Mapping ───────────────────────────────

HF_DATASETS = {
    "medical": [
        {
            "dataset": "bigbio/med_qa",
            "config": None,
            "split": "train",
            "text_fields": ["question", "answer"],
            "max_samples": 5000,
            "description": "Medical QA from professional exams",
        },
        {
            "dataset": "gamino/wiki_medical_terms",
            "config": None,
            "split": "train",
            "text_fields": ["page_text"],
            "max_samples": 5000,
            "description": "Wikipedia medical term definitions",
        },
        {
            "dataset": "BI55/MedText",
            "config": None,
            "split": "train",
            "text_fields": ["medical_abstract"],
            "max_samples": 5000,
            "description": "Medical abstracts for text classification",
        },
        {
            "dataset": "pubmed_qa",
            "config": "pqa_labeled",
            "split": "train",
            "text_fields": ["question", "long_answer"],
            "max_samples": 3000,
            "description": "PubMed question answering",
        },
    ],
    "legal": [
        {
            "dataset": "pile-of-law/pile-of-law",
            "config": "courtlistener_docket_entry_documents",
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 5000,
            "description": "Court docket entries and legal documents",
            "streaming": True,
        },
        {
            "dataset": "lexlms/lex_glue",
            "config": "eurlex",
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 5000,
            "description": "EU legal documents for NLU",
        },
        {
            "dataset": "nguha/legalbench",
            "config": "contract_nli_explicit_identification",
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 3000,
            "description": "Legal reasoning benchmark tasks",
        },
    ],
    "financial": [
        {
            "dataset": "financial_phrasebank",
            "config": "sentences_allagree",
            "split": "train",
            "text_fields": ["sentence"],
            "max_samples": 5000,
            "description": "Financial phrase sentiment analysis",
        },
        {
            "dataset": "AdaptLLM/finance-tasks",
            "config": None,
            "split": "train",
            "text_fields": ["input", "output"],
            "max_samples": 5000,
            "description": "Finance domain NLP tasks",
        },
        {
            "dataset": "zeroshot/twitter-financial-news-sentiment",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 3000,
            "description": "Financial news tweets with sentiment",
        },
    ],
    "scientific": [
        {
            "dataset": "scientific_papers",
            "config": "arxiv",
            "split": "train",
            "text_fields": ["abstract", "article"],
            "max_samples": 5000,
            "description": "ArXiv scientific paper abstracts",
            "streaming": True,
        },
        {
            "dataset": "allenai/scirepeval",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 3000,
            "description": "Scientific document evaluation",
        },
        {
            "dataset": "ccdv/pubmed-summarization",
            "config": None,
            "split": "train",
            "text_fields": ["abstract", "article"],
            "max_samples": 3000,
            "description": "PubMed article summarization",
            "streaming": True,
        },
    ],
    "engineering": [
        {
            "dataset": "bigcode/the-stack",
            "config": "data/python",
            "split": "train",
            "text_fields": ["content"],
            "max_samples": 3000,
            "description": "Source code from The Stack",
            "streaming": True,
        },
        {
            "dataset": "codeparrot/github-code",
            "config": "Python-all",
            "split": "train",
            "text_fields": ["code"],
            "max_samples": 3000,
            "description": "GitHub code repositories",
            "streaming": True,
        },
    ],
    "education": [
        {
            "dataset": "cais/mmlu",
            "config": "all",
            "split": "test",
            "text_fields": ["question"],
            "max_samples": 5000,
            "description": "Massive multitask language understanding",
        },
        {
            "dataset": "race",
            "config": "all",
            "split": "train",
            "text_fields": ["article", "question"],
            "max_samples": 5000,
            "description": "Reading comprehension from exams",
        },
        {
            "dataset": "openbookqa",
            "config": "main",
            "split": "train",
            "text_fields": ["question_stem", "fact1"],
            "max_samples": 3000,
            "description": "Science questions requiring reasoning",
        },
    ],
    "government": [
        {
            "dataset": "joelito/Multi_Legal_Pile",
            "config": "en",
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 5000,
            "description": "Multi-jurisdictional legal documents",
            "streaming": True,
        },
    ],
    "retail": [
        {
            "dataset": "katanaml/invoices-donut-data-v1",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 3000,
            "description": "Invoice document extraction",
        },
        {
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "config": "raw_review_All_Beauty",
            "split": "full",
            "text_fields": ["text", "title"],
            "max_samples": 5000,
            "description": "Amazon product reviews",
            "streaming": True,
        },
    ],
    "general": [
        {
            "dataset": "wikitext",
            "config": "wikitext-103-v1",
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 5000,
            "description": "Wikipedia articles (WikiText-103)",
        },
        {
            "dataset": "bookcorpus",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 5000,
            "description": "Book corpus for general language",
            "streaming": True,
        },
        {
            "dataset": "cc_news",
            "config": None,
            "split": "train",
            "text_fields": ["text", "title"],
            "max_samples": 5000,
            "description": "Common Crawl news articles",
            "streaming": True,
        },
    ],
    "multilingual": [
        {
            "dataset": "facebook/flores",
            "config": "all",
            "split": "dev",
            "text_fields": ["sentence"],
            "max_samples": 5000,
            "description": "Multilingual translation benchmark",
        },
        {
            "dataset": "papluca/language-identification",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 5000,
            "description": "Multi-language text identification",
        },
    ],
    "handwriting": [
        {
            "dataset": "Teklia/IAM-line",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "max_samples": 3000,
            "description": "IAM handwriting line recognition",
        },
    ],
}


# ─── Text Processing ────────────────────────────────────────────────────

def clean_text(text):
    """Clean and normalize text for teaching."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,;:!?\'"()\-\[\]{}/@#$%&*+=<>]', '', text)
    return text


def extract_sentences(text, min_length=10, max_length=500):
    """Split text into sentences suitable for teaching."""
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if min_length <= len(s.strip()) <= max_length]


def generate_ocr_pairs(text, domain_confusions=None):
    """Generate synthetic OCR confusion pairs."""
    base_confusions = {
        'O': '0', '0': 'O', 'l': '1', '1': 'l',
        'S': '5', '5': 'S', 'B': '8', '8': 'B',
        'n': 'r', 'e': 'c', 'h': 'b', 'a': 'o',
    }
    if domain_confusions:
        base_confusions.update(domain_confusions)

    pairs = []
    words = text.split()
    for word in words:
        if len(word) < 3:
            continue
        corrupted = list(word)
        applied = 0
        for i, char in enumerate(corrupted):
            if char in base_confusions and applied < 2:
                h = hashlib.md5(f"{word}{i}{len(pairs)}".encode()).hexdigest()
                if int(h[:2], 16) < 77:
                    corrupted[i] = base_confusions[char]
                    applied += 1
        result = ''.join(corrupted)
        if result != word:
            pairs.append({
                "input": result,
                "expected": word,
                "confidence": round(0.3 + 0.4 * (1 - applied / max(len(word), 1)), 3),
            })
    return pairs


# ─── HuggingFace Fetcher ────────────────────────────────────────────────

class HuggingFaceFetcher:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir

    def _load_dataset(self, ds_config):
        """Load a HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: 'datasets' package not installed. Run: pip install datasets")
            sys.exit(1)

        dataset_name = ds_config["dataset"]
        config = ds_config.get("config")
        split = ds_config.get("split", "train")
        streaming = ds_config.get("streaming", False)
        max_samples = ds_config.get("max_samples", 5000)

        print(f"  Loading: {dataset_name} (config={config}, split={split})")

        try:
            kwargs = {"path": dataset_name, "split": split}
            if config:
                kwargs["name"] = config
            if streaming:
                kwargs["streaming"] = True
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir

            ds = load_dataset(**kwargs)

            if streaming:
                # For streaming datasets, take only max_samples
                items = []
                for i, item in enumerate(ds):
                    if i >= max_samples:
                        break
                    items.append(item)
                return items
            else:
                # For regular datasets, slice
                if len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                return list(ds)

        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            return []

    def extract_texts(self, items, text_fields, max_samples=5000):
        """Extract text from dataset items."""
        texts = []
        for item in items:
            for field in text_fields:
                value = item.get(field)
                if isinstance(value, str) and len(value.strip()) > 5:
                    texts.append(value.strip())
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and len(v.strip()) > 5:
                            texts.append(v.strip())
            if len(texts) >= max_samples:
                break
        return texts[:max_samples]

    def process_domain(self, domain, output_dir, force=False):
        """Fetch and process all HuggingFace datasets for a domain."""
        if domain not in HF_DATASETS:
            print(f"Unknown domain: {domain}")
            return None

        datasets = HF_DATASETS[domain]
        all_sentences = []
        all_pairs = []
        datasets_processed = 0

        for ds_config in datasets:
            print(f"\nProcessing: {ds_config['dataset']}")

            # Check cache
            output_path = Path(output_dir)
            cache_file = output_path / f".cache_hf_{domain}_{ds_config['dataset'].replace('/', '_')}.json"

            if cache_file.exists() and not force:
                print(f"  Using cache: {cache_file.name}")
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    all_sentences.extend(cached.get("sentences", []))
                    all_pairs.extend(cached.get("pairs", []))
                    datasets_processed += 1
                    continue
                except Exception:
                    pass

            items = self._load_dataset(ds_config)
            if not items:
                print(f"  No data loaded from {ds_config['dataset']}")
                continue

            texts = self.extract_texts(items, ds_config["text_fields"],
                                       ds_config.get("max_samples", 5000))

            sentences = []
            pairs = []
            for text in texts:
                cleaned = clean_text(text)
                sents = extract_sentences(cleaned)
                sentences.extend(sents)
                p = generate_ocr_pairs(cleaned)
                pairs.extend(p)

            # Cache intermediate results
            output_path.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_file, 'w') as f:
                    json.dump({"sentences": sentences[:5000], "pairs": pairs[:2000]}, f)
            except Exception:
                pass

            all_sentences.extend(sentences)
            all_pairs.extend(pairs)
            datasets_processed += 1

            print(f"  Got {len(sentences)} sentences, {len(pairs)} pairs")

        # Deduplicate
        all_sentences = list(dict.fromkeys(all_sentences))

        # Save output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"huggingface_{domain}_seed.json"
        result = {
            "source": "huggingface",
            "domain": domain,
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "stats": {
                "total_sentences": len(all_sentences),
                "total_pairs": len(all_pairs),
                "datasets_processed": datasets_processed,
            },
            "sentences": all_sentences[:10000],
            "pairs": all_pairs[:5000],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {output_file}")
        print(f"  Sentences: {len(result['sentences'])}")
        print(f"  Pairs: {len(result['pairs'])}")

        return result


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shifu HuggingFace Data Seeder")
    parser.add_argument("--domain", type=str, help="Domain to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all domains")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--output", type=str, default="seeds_out",
                       help="Output directory (default: seeds_out)")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--cache-dir", type=str, help="HuggingFace cache directory")
    args = parser.parse_args()

    if args.list:
        print("\n=== Available HuggingFace Datasets by Domain ===\n")
        for domain, datasets in sorted(HF_DATASETS.items()):
            print(f"  {domain}:")
            for ds in datasets:
                streaming = " [streaming]" if ds.get("streaming") else ""
                print(f"    - {ds['dataset']}: {ds['description']}{streaming}")
            print()
        total = sum(len(v) for v in HF_DATASETS.values())
        print(f"Total: {total} datasets across {len(HF_DATASETS)} domains")
        return

    fetcher = HuggingFaceFetcher(cache_dir=args.cache_dir)

    if args.all:
        for domain in HF_DATASETS:
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
