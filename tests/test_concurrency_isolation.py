"""
tests/test_concurrency_isolation.py
Concurrency safety and process isolation tests.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.
"""

import json
import os
import shutil
import tempfile
import unittest

from campaign.run_spec import RunSpecification
from campaign.scheduler import CampaignScheduler
from campaign.artifacts import get_artifact_directory, read_metrics, read_status


class TestConcurrencyIsolation(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.scheduler = CampaignScheduler(
            base_results_dir=self.tmp_dir,
            max_workers=2,
            max_retries=0,
            allow_dirty=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_concurrent_runs_path_isolation(self):
        """Verify that concurrent runs have distinct non-colliding artifact paths."""
        spec1 = RunSpecification(
            block="B1",
            purpose="Concurrency test 1",
            dataset="nslkdd",
            strategy="tvflids",
            attack="clean",
            seed=42,
            num_rounds=1,
        )
        spec2 = RunSpecification(
            block="B1",
            purpose="Concurrency test 2",
            dataset="nslkdd",
            strategy="fedavg",
            attack="clean",
            seed=123,
            num_rounds=1,
        )

        dir1 = get_artifact_directory(spec1, base_dir=self.tmp_dir)
        dir2 = get_artifact_directory(spec2, base_dir=self.tmp_dir)

        self.assertNotEqual(spec1.run_id, spec2.run_id)
        self.assertNotEqual(dir1, dir2)

        # Run dry-run concurrently
        res = self.scheduler.schedule([spec1, spec2], dry_run=True, verbose=False)
        self.assertEqual(res["total"], 2)
        self.assertEqual(res["dry_run_skipped"], 2)

        # Confirm distinct status files written
        st1 = read_status(dir1)
        st2 = read_status(dir2)
        self.assertIsNotNone(st1)
        self.assertIsNotNone(st2)
        self.assertEqual(st1["status"], "SKIPPED")
        self.assertEqual(st2["status"], "SKIPPED")


if __name__ == "__main__":
    unittest.main()
