"""CPU preflight boundaries; no native model or tokenizer execution needed."""
import json
from pathlib import Path
import tempfile
import unittest

from .preflight import SPLITS, public_corpus
from .test_catalog_fusion import fixture


class PreflightTests(unittest.TestCase):
    def write_public(self, root):
        request, catalog = fixture()
        (root / "catalog.json").write_text(json.dumps({"products": list(catalog.values())}))
        (root / "public").mkdir()
        for split in SPLITS:
            (root / "public" / f"{split}.jsonl").write_text(json.dumps(request) + "\n")
        # Invalid label content makes accidental label parsing fail loudly.
        (root / "labels").mkdir()
        (root / "labels/train.jsonl").write_text("never parse labels during public preflight")
        return request, catalog

    def test_public_corpus_keeps_members_external_and_does_not_read_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_public(root)
            rows, splits, hashes = public_corpus(root)
            self.assertEqual(len(rows), 10)  # six variants plus four query rows
            self.assertEqual(set(splits), set(SPLITS))
            self.assertEqual(set(hashes), {"catalog.json", *(f"public/{s}.jsonl" for s in SPLITS)})
            for row in rows:
                self.assertNotIn("opaque", row["text"])
                self.assertNotIn("label", row["text"])

    def test_public_split_symlink_redirection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_public(root)
            path = root / "public/train.jsonl"
            path.unlink()
            path.symlink_to(root / "labels/train.jsonl")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                public_corpus(root)

    def test_unknown_candidate_is_rejected_before_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, _ = self.write_public(root)
            request["candidateProductIds"].append("unknown")
            (root / "public/train.jsonl").write_text(json.dumps(request) + "\n")
            with self.assertRaisesRegex(ValueError, "Unknown public candidate"):
                public_corpus(root)


if __name__ == "__main__":
    unittest.main()
