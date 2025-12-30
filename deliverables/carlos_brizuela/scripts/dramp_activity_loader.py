# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DRAMP Antimicrobial Peptide Activity Loader.

Downloads and processes antimicrobial peptide data from DRAMP database
for training activity predictors.

DRAMP: Data Repository of Antimicrobial Peptides
URL: http://dramp.cpu-bioinfor.org/

Usage:
    python dramp_activity_loader.py --download
    python dramp_activity_loader.py --train
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import csv
from typing import Optional
from dataclasses import dataclass, field, asdict
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

from shared.config import get_config
from shared.constants import HYDROPHOBICITY, CHARGES, VOLUMES, WHO_CRITICAL_PATHOGENS

# Try to import requests for downloading
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import sklearn for training
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, roc_auc_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class AMPRecord:
    """Container for an antimicrobial peptide record."""
    dramp_id: str
    sequence: str
    name: Optional[str] = None
    length: int = 0
    source: Optional[str] = None
    target_organism: Optional[str] = None
    mic_value: Optional[float] = None  # Minimum Inhibitory Concentration (μg/mL)
    mic_unit: str = "μg/mL"
    activity_type: Optional[str] = None
    hemolytic: Optional[float] = None  # HC50 or hemolysis %

    def __post_init__(self):
        self.length = len(self.sequence)

    def compute_features(self) -> dict:
        """Compute sequence features for ML."""
        seq = self.sequence.upper()
        n = len(seq)

        if n == 0:
            return {}

        # Amino acid composition
        aac = {aa: seq.count(aa) / n for aa in "ACDEFGHIKLMNPQRSTVWY"}

        # Physicochemical properties
        charge = sum(CHARGES.get(aa, 0) for aa in seq)
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / n
        volume = sum(VOLUMES.get(aa, 100) for aa in seq)

        # Compositional features
        positive = sum(1 for aa in seq if aa in "KRH") / n
        negative = sum(1 for aa in seq if aa in "DE") / n
        aromatic = sum(1 for aa in seq if aa in "FWY") / n
        aliphatic = sum(1 for aa in seq if aa in "AILV") / n
        polar = sum(1 for aa in seq if aa in "STNQ") / n

        # Amphipathicity (simplified)
        hydro_moment = 0
        if n >= 7:
            for i in range(n - 6):
                window = seq[i:i + 7]
                h_values = [HYDROPHOBICITY.get(aa, 0) for aa in window]
                hydro_moment += abs(sum(h_values))
            hydro_moment /= (n - 6)

        return {
            "length": n,
            "charge": charge,
            "hydrophobicity": hydro,
            "volume": volume,
            "positive_fraction": positive,
            "negative_fraction": negative,
            "aromatic_fraction": aromatic,
            "aliphatic_fraction": aliphatic,
            "polar_fraction": polar,
            "hydrophobic_moment": hydro_moment,
            **{f"aac_{aa}": v for aa, v in aac.items()}
        }


@dataclass
class AMPDatabase:
    """Database of antimicrobial peptides with activity data."""
    records: list[AMPRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_record(self, record: AMPRecord):
        """Add a peptide record."""
        self.records.append(record)

    def filter_by_target(self, target: str) -> list[AMPRecord]:
        """Filter records by target organism."""
        target_lower = target.lower()
        return [r for r in self.records
                if r.target_organism and target_lower in r.target_organism.lower()]

    def filter_by_mic(self, max_mic: float = 100) -> list[AMPRecord]:
        """Filter records with MIC below threshold."""
        return [r for r in self.records
                if r.mic_value is not None and r.mic_value <= max_mic]

    def get_training_data(self, target: str = None) -> tuple[np.ndarray, np.ndarray]:
        """Get features and labels for ML training.

        Args:
            target: Optional target organism filter

        Returns:
            X (features), y (log10 MIC values)
        """
        records = self.filter_by_target(target) if target else self.records
        records = [r for r in records if r.mic_value is not None and r.mic_value > 0]

        if not records:
            return np.array([]), np.array([])

        features = []
        labels = []

        for record in records:
            feat = record.compute_features()
            if feat:
                features.append(list(feat.values()))
                labels.append(np.log10(record.mic_value))

        return np.array(features), np.array(labels)

    def save(self, path: Path):
        """Save database to JSON."""
        data = {
            "metadata": self.metadata,
            "records": [asdict(r) for r in self.records]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AMPDatabase":
        """Load database from JSON."""
        with open(path) as f:
            data = json.load(f)

        db = cls(metadata=data.get("metadata", {}))
        for rec_data in data.get("records", []):
            db.add_record(AMPRecord(**rec_data))
        return db


class DRAMPLoader:
    """Load antimicrobial peptide data from DRAMP database."""

    DRAMP_URLS = {
        "general": "http://dramp.cpu-bioinfor.org/downloads/download_data/DRAMP_general_amps.csv",
        "patent": "http://dramp.cpu-bioinfor.org/downloads/download_data/DRAMP_patent_amps.csv",
    }

    # Alternative: curated demo data
    DEMO_PEPTIDES = [
        # Known AMPs with activity data
        ("Magainin 2", "GIGKFLHSAKKFGKAFVGEIMNS", "Escherichia coli", 10.0),
        ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ", "Staphylococcus aureus", 2.0),
        ("LL-37", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", "Pseudomonas aeruginosa", 4.0),
        ("Cecropin A", "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "Escherichia coli", 0.5),
        ("Defensin HNP-1", "ACYCRIPACIAGERRYGTCIYQGRLWAFCC", "Staphylococcus aureus", 5.0),
        ("Indolicidin", "ILPWKWPWWPWRR", "Escherichia coli", 8.0),
        ("Protegrin-1", "RGGRLCYCRRRFCVCVGR", "Pseudomonas aeruginosa", 1.0),
        ("Lactoferricin B", "FKCRRWQWRMKKLGAPSITCVRRAF", "Escherichia coli", 4.0),
        ("Pexiganan", "GIGKFLKKAKKFGKAFVKILKK", "Acinetobacter baumannii", 4.0),
        ("Polymyxin B", "MDRBTBBBFBBLBT", "Acinetobacter baumannii", 0.5),  # Simplified
    ]

    def __init__(self):
        self.config = get_config()
        self.cache_dir = self.config.get_partner_dir("carlos") / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.config.get_partner_dir("carlos") / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_dramp(self, dataset: str = "general") -> Optional[str]:
        """Download DRAMP dataset.

        Args:
            dataset: Dataset name ("general" or "patent")

        Returns:
            Path to downloaded file or None
        """
        if not REQUESTS_AVAILABLE:
            print("Requests not available - using demo data")
            return None

        url = self.DRAMP_URLS.get(dataset)
        if not url:
            print(f"Unknown dataset: {dataset}")
            return None

        cache_path = self.cache_dir / f"dramp_{dataset}.csv"

        if cache_path.exists():
            print(f"Using cached {dataset} data")
            return str(cache_path)

        try:
            print(f"Downloading DRAMP {dataset} data...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(cache_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded to {cache_path}")
            return str(cache_path)

        except Exception as e:
            print(f"Error downloading DRAMP data: {e}")
            return None

    def parse_dramp_csv(self, csv_path: str) -> list[AMPRecord]:
        """Parse DRAMP CSV file to records."""
        records = []

        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract relevant fields (column names may vary)
                    sequence = row.get("Sequence", row.get("sequence", ""))
                    if not sequence or len(sequence) < 5:
                        continue

                    # Parse MIC value
                    mic_str = row.get("MIC", row.get("mic_value", ""))
                    mic_value = None
                    if mic_str:
                        try:
                            # Handle various formats: "10", "10 μg/mL", ">100"
                            mic_clean = mic_str.replace(">", "").replace("<", "")
                            mic_clean = mic_clean.split()[0]
                            mic_value = float(mic_clean)
                        except (ValueError, IndexError):
                            pass

                    record = AMPRecord(
                        dramp_id=row.get("DRAMP_ID", row.get("id", f"AMP_{len(records)}")),
                        sequence=sequence.upper().replace(" ", ""),
                        name=row.get("Name", row.get("name")),
                        source=row.get("Source", row.get("source")),
                        target_organism=row.get("Target_Organism", row.get("target")),
                        mic_value=mic_value,
                        activity_type=row.get("Activity", row.get("activity_type")),
                    )
                    records.append(record)

        except Exception as e:
            print(f"Error parsing CSV: {e}")

        return records

    def generate_demo_database(self) -> AMPDatabase:
        """Generate demo database with curated peptides."""
        db = AMPDatabase(metadata={
            "source": "Demo",
            "description": "Curated antimicrobial peptides for demonstration"
        })

        # Add known AMPs
        for i, (name, seq, target, mic) in enumerate(self.DEMO_PEPTIDES):
            record = AMPRecord(
                dramp_id=f"DEMO_{i + 1:04d}",
                sequence=seq,
                name=name,
                target_organism=target,
                mic_value=mic,
            )
            db.add_record(record)

        # Generate synthetic variants
        import random
        random.seed(42)

        aa_pool = "ACDEFGHIKLMNPQRSTVWY"
        targets = [
            "Acinetobacter baumannii",
            "Pseudomonas aeruginosa",
            "Klebsiella pneumoniae",
            "Staphylococcus aureus",
            "Escherichia coli",
        ]

        for i in range(200):
            # Generate peptide with AMP-like properties
            length = random.randint(12, 30)

            # Bias toward cationic and hydrophobic
            aa_weights = [1] * 20
            for j, aa in enumerate(aa_pool):
                if aa in "KRH":
                    aa_weights[j] = 3  # More cationic
                elif aa in "ILVFWY":
                    aa_weights[j] = 2  # More hydrophobic

            seq = "".join(random.choices(aa_pool, weights=aa_weights, k=length))

            # Compute crude activity prediction
            charge = sum(CHARGES.get(aa, 0) for aa in seq)
            hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / length

            # MIC correlates with charge and hydrophobicity (simplified model)
            base_mic = 32
            if charge >= 4:
                base_mic /= 2
            if charge >= 6:
                base_mic /= 2
            if 0 < hydro < 1:
                base_mic /= 2
            if hydro > 1.5:
                base_mic *= 2  # Too hydrophobic

            mic = base_mic * (0.5 + random.random())  # Add noise

            record = AMPRecord(
                dramp_id=f"SYNTH_{i + 1:04d}",
                sequence=seq,
                name=f"Synthetic AMP {i + 1}",
                target_organism=random.choice(targets),
                mic_value=round(mic, 1),
            )
            db.add_record(record)

        return db

    def load_or_download(
        self,
        cache_name: str = "amp_database.json",
        force_download: bool = False,
    ) -> AMPDatabase:
        """Load from cache or download.

        Args:
            cache_name: Cache filename
            force_download: Force re-download

        Returns:
            AMPDatabase
        """
        cache_path = self.cache_dir / cache_name

        if cache_path.exists() and not force_download:
            print(f"Loading cached database from {cache_path}")
            return AMPDatabase.load(cache_path)

        # Try to download DRAMP
        csv_path = self.download_dramp("general")

        if csv_path:
            print("Parsing DRAMP data...")
            records = self.parse_dramp_csv(csv_path)
            db = AMPDatabase(
                records=records,
                metadata={"source": "DRAMP", "records_count": len(records)}
            )
        else:
            print("Using demo database...")
            db = self.generate_demo_database()

        db.save(cache_path)
        return db

    def train_activity_predictor(
        self,
        db: AMPDatabase,
        target: str = None,
        model_name: str = "activity_predictor",
    ) -> Optional[dict]:
        """Train an activity predictor on the database.

        Args:
            db: AMPDatabase with activity data
            target: Target organism (None = all)
            model_name: Name for saved model

        Returns:
            Training metrics or None if failed
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available for training")
            return None

        X, y = db.get_training_data(target)

        if len(X) < 20:
            print(f"Not enough training data: {len(X)} samples")
            return None

        print(f"Training on {len(X)} samples...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Correlation
        from scipy.stats import pearsonr
        r, p = pearsonr(y_test, y_pred)

        metrics = {
            "n_samples": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "rmse": float(rmse),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "target": target or "all",
        }

        print(f"  RMSE: {rmse:.3f}")
        print(f"  Pearson r: {r:.3f}")

        # Save model
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump({"model": model, "scaler": scaler, "metrics": metrics}, model_path)
        print(f"  Saved model to {model_path}")

        return metrics

    def train_all_pathogen_models(self, db: AMPDatabase) -> dict:
        """Train activity predictors for all WHO critical pathogens.

        Args:
            db: AMPDatabase

        Returns:
            Dictionary of metrics per pathogen
        """
        all_metrics = {}

        # Train general model first
        metrics = self.train_activity_predictor(db, target=None, model_name="activity_general")
        if metrics:
            all_metrics["general"] = metrics

        # Train pathogen-specific models
        pathogens = [
            ("acinetobacter", "Acinetobacter baumannii"),
            ("pseudomonas", "Pseudomonas aeruginosa"),
            ("staphylococcus", "Staphylococcus aureus"),
            ("escherichia", "Escherichia coli"),
        ]

        for short_name, full_name in pathogens:
            print(f"\nTraining model for {full_name}...")
            metrics = self.train_activity_predictor(
                db,
                target=short_name,
                model_name=f"activity_{short_name}"
            )
            if metrics:
                all_metrics[short_name] = metrics

        return all_metrics


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and train on DRAMP AMP data")
    parser.add_argument("--download", action="store_true", help="Download DRAMP data")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--train", action="store_true", help="Train activity predictors")
    parser.add_argument("--list", action="store_true", help="List database statistics")
    args = parser.parse_args()

    loader = DRAMPLoader()

    if args.download or args.force:
        db = loader.load_or_download(force_download=args.force)
        print(f"\nLoaded {len(db.records)} peptide records")

        if args.train:
            print("\n" + "=" * 50)
            print("Training Activity Predictors")
            print("=" * 50)
            metrics = loader.train_all_pathogen_models(db)
            print("\nTraining Summary:")
            for name, m in metrics.items():
                print(f"  {name}: r={m['pearson_r']:.3f}, RMSE={m['rmse']:.3f}, n={m['n_samples']}")

    elif args.list:
        cache_path = loader.cache_dir / "amp_database.json"
        if cache_path.exists():
            db = AMPDatabase.load(cache_path)
            print(f"Database: {len(db.records)} total records")

            # Count by target
            targets = {}
            for r in db.records:
                if r.target_organism:
                    key = r.target_organism.split()[0]  # First word
                    targets[key] = targets.get(key, 0) + 1

            print("\nBy target organism:")
            for target, count in sorted(targets.items(), key=lambda x: -x[1])[:10]:
                print(f"  {target}: {count}")
        else:
            print("No cached database. Run with --download first.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
