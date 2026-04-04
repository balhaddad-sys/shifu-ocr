"""
Shifu Unified Seed Registry
Orchestrates data seeding from all sources (Kaggle, HuggingFace, custom files)
into a unified format for the teaching model.

Usage:
    python -m teaching.seeds.seed_registry --domain medical --output seeds_out/
    python -m teaching.seeds.seed_registry --all --output seeds_out/
    python -m teaching.seeds.seed_registry --ingest seeds_out/ --emit teaching_data.json
    python -m teaching.seeds.seed_registry --stats seeds_out/

This is the main entry point for the seeding pipeline.
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime

# ─── Seed Sources ────────────────────────────────────────────────────────

ALL_DOMAINS = [
    "medical", "legal", "financial", "scientific", "engineering",
    "education", "government", "retail", "logistics", "general",
    "multilingual", "handwriting",
]

# Built-in corpus seeds (no external download needed)
BUILTIN_SEEDS = {
    "medical": {
        "sentences": [
            "Patient admitted with acute myocardial infarction requiring emergent catheterization.",
            "Nurse administered morphine sulfate 4mg IV for chest pain management.",
            "Laboratory results show elevated troponin I at 2.4 ng/mL.",
            "Physician ordered continuous cardiac monitoring and serial ECGs.",
            "Blood pressure 140/90 mmHg, heart rate 98 bpm, respiratory rate 22.",
            "Chest X-ray reveals bilateral pulmonary infiltrates consistent with pneumonia.",
            "Patient started on piperacillin-tazobactam 4.5g IV every 6 hours.",
            "Wound culture grew methicillin-resistant Staphylococcus aureus.",
            "Neurology consult recommended MRI brain with contrast.",
            "Discharge summary completed with follow-up in cardiology clinic.",
        ],
        "pairs": [
            {"input": "m0rphine", "expected": "morphine", "confidence": 0.4},
            {"input": "tr0ponin", "expected": "troponin", "confidence": 0.3},
            {"input": "piperacil1in", "expected": "piperacillin", "confidence": 0.4},
            {"input": "taz0bactam", "expected": "tazobactam", "confidence": 0.3},
            {"input": "card1ac", "expected": "cardiac", "confidence": 0.5},
        ],
    },
    "legal": {
        "sentences": [
            "The plaintiff alleges breach of contract under Section 2-207 of the UCC.",
            "Defendant filed a motion to dismiss for failure to state a claim.",
            "The court granted summary judgment in favor of the respondent.",
            "Counsel submitted memorandum of law in support of preliminary injunction.",
            "Pursuant to Rule 12(b)(6), the complaint fails to state actionable claims.",
            "The arbitration clause in paragraph 14.3 governs dispute resolution.",
            "Witness testified under oath regarding the chain of custody.",
            "The statute of limitations for tort claims is three years from discovery.",
        ],
        "pairs": [
            {"input": "p1aintiff", "expected": "plaintiff", "confidence": 0.4},
            {"input": "ju0gment", "expected": "judgment", "confidence": 0.3},
            {"input": "arb1tration", "expected": "arbitration", "confidence": 0.4},
        ],
    },
    "financial": {
        "sentences": [
            "Quarterly revenue increased 12% year-over-year to $4.2 billion.",
            "The Federal Reserve raised interest rates by 25 basis points.",
            "Net income attributable to shareholders was $1.8 million.",
            "Operating expenses decreased due to restructuring charges of $500K.",
            "Earnings per share came in at $2.15, beating consensus estimates.",
            "The portfolio allocation shifted toward fixed-income securities.",
            "Accounts receivable turnover ratio improved to 8.2 times.",
            "Depreciation and amortization totaled $340 million for the quarter.",
        ],
        "pairs": [
            {"input": "$4.2 bi11ion", "expected": "$4.2 billion", "confidence": 0.3},
            {"input": "$1.8 mi1lion", "expected": "$1.8 million", "confidence": 0.3},
            {"input": "depreciati0n", "expected": "depreciation", "confidence": 0.4},
        ],
    },
    "scientific": {
        "sentences": [
            "The experiment demonstrated a statistically significant correlation (p < 0.01).",
            "Spectroscopic analysis revealed absorption peaks at 254nm and 380nm.",
            "The catalyst increased reaction yield from 45% to 92% under mild conditions.",
            "Genome sequencing identified three novel single nucleotide polymorphisms.",
            "The thermal conductivity of the composite material was 0.35 W/mK.",
            "Results were reproducible across five independent trials with SD < 0.05.",
        ],
        "pairs": [
            {"input": "cata1yst", "expected": "catalyst", "confidence": 0.4},
            {"input": "p0lymorphisms", "expected": "polymorphisms", "confidence": 0.3},
            {"input": "spectr0scopic", "expected": "spectroscopic", "confidence": 0.3},
        ],
    },
    "engineering": {
        "sentences": [
            "The microcontroller operates at 3.3V with a clock frequency of 168MHz.",
            "Tensile strength of the alloy exceeded 450 MPa at room temperature.",
            "The control loop uses PID with gains Kp=2.5, Ki=0.8, Kd=0.3.",
            "CAD model exported as STEP file for CNC machining verification.",
            "Load testing revealed maximum deflection of 2.3mm under 500N.",
            "The firmware update resolved the I2C communication timeout issue.",
        ],
        "pairs": [
            {"input": "micr0controller", "expected": "microcontroller", "confidence": 0.4},
            {"input": "def1ection", "expected": "deflection", "confidence": 0.4},
        ],
    },
    "general": {
        "sentences": [
            "The conference attracted over three thousand attendees from forty countries.",
            "Chapter twelve explores the historical context of the industrial revolution.",
            "The editorial board reviewed manuscripts from leading international scholars.",
            "Public transportation services were temporarily suspended during the storm.",
            "The documentary film received critical acclaim at the international festival.",
        ],
        "pairs": [],
    },
}


# ─── Registry ───────────────────────────────────────────────────────────

class SeedRegistry:
    def __init__(self, output_dir="seeds_out"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {}

    def seed_builtin(self, domain):
        """Export built-in seed data for a domain."""
        if domain not in BUILTIN_SEEDS:
            return None

        seed = BUILTIN_SEEDS[domain]
        output_file = self.output_dir / f"builtin_{domain}_seed.json"

        result = {
            "source": "builtin",
            "domain": domain,
            "generated_at": datetime.now().isoformat(),
            "stats": {
                "total_sentences": len(seed["sentences"]),
                "total_pairs": len(seed["pairs"]),
            },
            "sentences": seed["sentences"],
            "pairs": seed["pairs"],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.manifest[f"builtin_{domain}"] = {
            "file": str(output_file),
            "domain": domain,
            "source": "builtin",
            "sentences": len(seed["sentences"]),
            "pairs": len(seed["pairs"]),
        }

        return result

    def seed_kaggle(self, domain, force=False):
        """Fetch and process Kaggle data for a domain."""
        try:
            from teaching.seeds.kaggle import KaggleFetcher
        except ImportError:
            try:
                from kaggle import KaggleFetcher
            except ImportError:
                print(f"  Skipping Kaggle for {domain} (module not available)")
                return None

        fetcher = KaggleFetcher()
        result = fetcher.process_domain(domain, str(self.output_dir), force=force)
        if result:
            self.manifest[f"kaggle_{domain}"] = {
                "file": str(self.output_dir / f"kaggle_{domain}_seed.json"),
                "domain": domain,
                "source": "kaggle",
                "sentences": result["stats"]["total_sentences"],
                "pairs": result["stats"]["total_pairs"],
            }
        return result

    def seed_huggingface(self, domain, force=False):
        """Fetch and process HuggingFace data for a domain."""
        try:
            from teaching.seeds.huggingface import HuggingFaceFetcher
        except ImportError:
            try:
                from huggingface import HuggingFaceFetcher
            except ImportError:
                print(f"  Skipping HuggingFace for {domain} (module not available)")
                return None

        fetcher = HuggingFaceFetcher()
        result = fetcher.process_domain(domain, str(self.output_dir), force=force)
        if result:
            self.manifest[f"huggingface_{domain}"] = {
                "file": str(self.output_dir / f"huggingface_{domain}_seed.json"),
                "domain": domain,
                "source": "huggingface",
                "sentences": result["stats"]["total_sentences"],
                "pairs": result["stats"]["total_pairs"],
            }
        return result

    def seed_domain(self, domain, sources=None, force=False):
        """Seed a domain from all available sources."""
        if sources is None:
            sources = ["builtin", "kaggle", "huggingface"]

        results = {}
        print(f"\n{'='*60}")
        print(f"  Seeding domain: {domain}")
        print(f"{'='*60}")

        if "builtin" in sources:
            r = self.seed_builtin(domain)
            if r:
                results["builtin"] = r
                print(f"  Builtin: {r['stats']['total_sentences']} sentences")

        if "kaggle" in sources:
            r = self.seed_kaggle(domain, force=force)
            if r:
                results["kaggle"] = r
                print(f"  Kaggle: {r['stats']['total_sentences']} sentences")

        if "huggingface" in sources:
            r = self.seed_huggingface(domain, force=force)
            if r:
                results["huggingface"] = r
                print(f"  HuggingFace: {r['stats']['total_sentences']} sentences")

        return results

    def seed_all(self, sources=None, force=False):
        """Seed all domains from all sources."""
        for domain in ALL_DOMAINS:
            self.seed_domain(domain, sources=sources, force=force)
        self.save_manifest()

    def ingest_seeds(self, seed_dir=None):
        """Read all seed files and merge into unified teaching data."""
        seed_dir = Path(seed_dir or self.output_dir)
        seed_files = list(seed_dir.glob("*_seed.json"))

        unified = {}
        for seed_file in seed_files:
            try:
                with open(seed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                domain = data.get("domain", "unknown")
                source = data.get("source", "unknown")

                if domain not in unified:
                    unified[domain] = {
                        "sentences": [],
                        "pairs": [],
                        "sources": [],
                    }

                unified[domain]["sentences"].extend(data.get("sentences", []))
                unified[domain]["pairs"].extend(data.get("pairs", []))
                unified[domain]["sources"].append({
                    "source": source,
                    "file": seed_file.name,
                    "sentences": len(data.get("sentences", [])),
                    "pairs": len(data.get("pairs", [])),
                })
            except Exception as e:
                print(f"  Warning: Could not read {seed_file.name}: {e}")

        # Deduplicate sentences per domain
        for domain in unified:
            unified[domain]["sentences"] = list(dict.fromkeys(unified[domain]["sentences"]))

        return unified

    def emit_teaching_data(self, output_file, seed_dir=None):
        """Merge all seeds and emit unified teaching data JSON."""
        unified = self.ingest_seeds(seed_dir)

        result = {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "domains": {},
            "summary": {
                "total_domains": len(unified),
                "total_sentences": 0,
                "total_pairs": 0,
            },
        }

        for domain, data in unified.items():
            result["domains"][domain] = {
                "sentences": data["sentences"],
                "pairs": data["pairs"],
                "sources": data["sources"],
                "stats": {
                    "sentences": len(data["sentences"]),
                    "pairs": len(data["pairs"]),
                },
            }
            result["summary"]["total_sentences"] += len(data["sentences"])
            result["summary"]["total_pairs"] += len(data["pairs"])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nEmitted teaching data: {output_file}")
        print(f"  Domains: {result['summary']['total_domains']}")
        print(f"  Total sentences: {result['summary']['total_sentences']}")
        print(f"  Total pairs: {result['summary']['total_pairs']}")

        return result

    def save_manifest(self):
        """Save the seed manifest."""
        manifest_file = self.output_dir / "seed_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "entries": self.manifest,
            }, f, indent=2)

    def print_stats(self, seed_dir=None):
        """Print statistics about available seeds."""
        unified = self.ingest_seeds(seed_dir)

        print(f"\n{'='*60}")
        print(f"  Shifu Seed Statistics")
        print(f"{'='*60}\n")

        total_sentences = 0
        total_pairs = 0

        for domain in sorted(unified.keys()):
            data = unified[domain]
            ns = len(data["sentences"])
            np_ = len(data["pairs"])
            total_sentences += ns
            total_pairs += np_
            sources = ", ".join(s["source"] for s in data["sources"])
            print(f"  {domain:15s}  {ns:>6} sentences  {np_:>5} pairs  [{sources}]")

        print(f"\n  {'TOTAL':15s}  {total_sentences:>6} sentences  {total_pairs:>5} pairs")
        print(f"  Domains: {len(unified)}")
        print()


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shifu Unified Seed Registry")
    parser.add_argument("--domain", type=str, help="Domain to seed")
    parser.add_argument("--all", action="store_true", help="Seed all domains")
    parser.add_argument("--builtin-only", action="store_true",
                       help="Only use built-in seeds (no downloads)")
    parser.add_argument("--output", type=str, default="seeds_out",
                       help="Output directory")
    parser.add_argument("--ingest", type=str,
                       help="Ingest seeds from directory and merge")
    parser.add_argument("--emit", type=str,
                       help="Emit unified teaching data to file")
    parser.add_argument("--stats", type=str, nargs='?', const="seeds_out",
                       help="Show statistics for seeds in directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    registry = SeedRegistry(output_dir=args.output)

    if args.stats is not None:
        registry.print_stats(args.stats)
        return

    if args.ingest and args.emit:
        registry.emit_teaching_data(args.emit, args.ingest)
        return

    sources = ["builtin"] if args.builtin_only else None

    if args.all:
        registry.seed_all(sources=sources, force=args.force)
    elif args.domain:
        registry.seed_domain(args.domain, sources=sources, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
