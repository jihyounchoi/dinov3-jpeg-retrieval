import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ads_dinov3.metrics import (
    bootstrap_ci,
    class_precision_at_k,
    clean_compressed_similarity,
    instance_recall_at_k,
    same_id_exclude_mask,
    summarize_retrieval,
    topk_search,
)
from ads_dinov3.retrieval import mine_failure_cases
from ads_dinov3.runlog import append_run_log, utc_now


class RetrievalMetricTest(unittest.TestCase):
    def setUp(self):
        self.gallery_features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.query_features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.gallery_ids = np.asarray(["a", "b", "c", "d"])
        self.query_ids = np.asarray(["a", "b"])
        self.gallery_classes = np.asarray([0, 1, 1, 2])
        self.query_classes = np.asarray([0, 1])

    def test_topk_and_instance_recall(self):
        result = topk_search(self.query_features, self.gallery_features, k=2)
        self.assertEqual(result.topk_indices.shape, (2, 2))
        self.assertAlmostEqual(instance_recall_at_k(result.topk_indices, self.query_ids, self.gallery_ids, 1), 1.0)

    def test_class_precision_include_original(self):
        result = topk_search(self.query_features, self.gallery_features, k=2)
        value = class_precision_at_k(result.topk_indices, self.query_classes, self.gallery_classes, 2)
        self.assertAlmostEqual(value, 0.75)

    def test_exclude_original_mask(self):
        mask = same_id_exclude_mask(self.query_ids, self.gallery_ids)
        result = topk_search(self.query_features, self.gallery_features, k=1, exclude_mask=mask)
        self.assertNotEqual(self.gallery_ids[result.topk_indices[0, 0]], "a")
        self.assertNotEqual(self.gallery_ids[result.topk_indices[1, 0]], "b")

    def test_summarize_retrieval_reports_both_class_policies(self):
        rows, _, _ = summarize_retrieval(
            self.query_features,
            self.query_ids,
            self.query_classes,
            self.gallery_features,
            self.gallery_ids,
            self.gallery_classes,
            ks=[1, 2],
            topk_for_export=2,
        )
        policies = {(row["metric"], row["self_policy"], row["k"]) for row in rows}
        self.assertIn(("class_precision", "include_original", 1), policies)
        self.assertIn(("class_precision", "exclude_original", 1), policies)
        self.assertIn(("instance_recall", "include_original", 1), policies)
        for row in rows:
            self.assertIn("ci_low", row)
            self.assertIn("ci_high", row)
            self.assertIn("bootstrap_B", row)
            self.assertIn("n_queries", row)

    def test_summarize_retrieval_with_bootstrap_returns_finite_ci(self):
        rng = np.random.default_rng(0)
        gallery = rng.standard_normal((40, 8)).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        query = gallery[:20] + 0.05 * rng.standard_normal((20, 8)).astype(np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        gallery_ids = np.asarray([f"id_{i}" for i in range(40)])
        query_ids = gallery_ids[:20]
        gallery_classes = np.asarray([i % 4 for i in range(40)])
        query_classes = gallery_classes[:20]
        rows, _, _ = summarize_retrieval(
            query, query_ids, query_classes,
            gallery, gallery_ids, gallery_classes,
            ks=[1, 5], topk_for_export=5,
            bootstrap_B=200, bootstrap_seed=2026, bootstrap_alpha=0.05,
        )
        for row in rows:
            self.assertGreaterEqual(row["ci_high"], row["value"] - 1e-6)
            self.assertLessEqual(row["ci_low"], row["value"] + 1e-6)
            self.assertEqual(row["bootstrap_B"], 200)

    def test_bootstrap_ci_is_deterministic_with_seed(self):
        per_query = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        lo1, hi1 = bootstrap_ci(per_query, B=500, seed=42, alpha=0.05)
        lo2, hi2 = bootstrap_ci(per_query, B=500, seed=42, alpha=0.05)
        self.assertEqual((lo1, hi1), (lo2, hi2))
        self.assertLessEqual(lo1, float(np.mean(per_query)))
        self.assertGreaterEqual(hi1, float(np.mean(per_query)))

    def test_clean_compressed_similarity_matches_image_id(self):
        values = clean_compressed_similarity(self.query_features, self.query_ids, self.gallery_features, self.gallery_ids)
        np.testing.assert_allclose(values, np.asarray([1.0, 1.0], dtype=np.float32))

    def test_mine_failure_cases_reports_severe_rank_shift(self):
        rows = []
        for model, rank in [("dinov3", 1), ("resnet50", -1), ("vit_b_16", 8)]:
            rows.append(
                {
                    "model": model,
                    "quality": 10,
                    "image_id": "img_001",
                    "query_class": 0,
                    "query_path": "query.jpg",
                    "instance_rank_within_exported_topk": rank,
                    "instance_hit_at_1": rank == 1,
                    "instance_hit_at_5": 1 <= rank <= 5,
                    "instance_hit_at_10": 1 <= rank <= 10,
                    "same_class_at_10": 1,
                    "top1_id": "img_001" if rank == 1 else "other",
                    "top1_class": 0,
                    "top1_path": "top1.jpg",
                    "top1_similarity": 0.9,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_cases.csv"
            mine_failure_cases(rows, path)
            with path.open("r", newline="", encoding="utf-8") as f:
                cases = list(csv.DictReader(f))
        self.assertTrue(any(row["case_type"] == "severe_compression_rank_shift" for row in cases))

    def test_mine_failure_cases_empty_file_keeps_threshold_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_cases.csv"
            mine_failure_cases([], path, rank_shift_threshold=5)
            with path.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
        self.assertEqual(header, ["case_type", "rank_shift_threshold"])

    def test_run_log_writes_seed_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "_project_root": tmp,
                "_config_path": "configs/default.yaml",
                "runtime": {"seed": 2026},
            }
            start = utc_now()
            append_run_log(
                cfg,
                start=start,
                end=utc_now(),
                run_name="unit",
                mode="sanity",
                status="success",
            )
            with (Path(tmp) / "outputs" / "run_log.csv").open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["seed"], "2026")


if __name__ == "__main__":
    unittest.main()
