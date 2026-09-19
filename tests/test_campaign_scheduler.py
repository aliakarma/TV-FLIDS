"""
tests/test_campaign_scheduler.py
Unit and functional tests for CampaignScheduler.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from campaign.run_spec import RunSpecification
from campaign.scheduler import CampaignScheduler
from campaign.artifacts import get_artifact_directory, write_status, write_metrics


class TestCampaignScheduler(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.scheduler = CampaignScheduler(
            base_results_dir=self.tmp_dir,
            max_workers=2,
            max_retries=1,
            allow_dirty=True,
        )

        self.spec_nslkdd = RunSpecification(
            block="B1",
            purpose="Tuning on NSL-KDD",
            dataset="nslkdd",
            strategy="tvflids",
            attack="clean",
            seed=42,
            num_rounds=2,
        )

        self.spec_ciciot = RunSpecification(
            block="B1",
            purpose="Tuning on CIC-IoT-2023",
            dataset="ciciot2023",
            strategy="tvflids",
            attack="clean",
            seed=42,
            num_rounds=2,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_discover_run_states(self):
        specs = [self.spec_nslkdd, self.spec_ciciot]
        discovered = self.scheduler.discover_run_states(specs)

        # spec_ciciot must be categorized as blocked
        self.assertEqual(len(discovered["blocked"]), 1)
        self.assertEqual(discovered["blocked"][0][0].dataset, "ciciot2023")

        # spec_nslkdd must be categorized as pending
        self.assertEqual(len(discovered["pending"]), 1)
        self.assertEqual(discovered["pending"][0][0].dataset, "nslkdd")

    def test_dry_run_scheduling(self):
        specs = [self.spec_nslkdd, self.spec_ciciot]
        res = self.scheduler.schedule(specs, dry_run=True, verbose=False)

        self.assertEqual(res["total"], 2)
        self.assertEqual(res["blocked"], 1)
        self.assertEqual(res["dry_run_skipped"], 1)
        self.assertEqual(res["failed"], 0)

        # Check status written on disk for blocked
        out_ciciot = get_artifact_directory(self.spec_ciciot, base_dir=self.tmp_dir)
        with open(os.path.join(out_ciciot, "status.json"), "r") as f:
            st = json.load(f)
        self.assertEqual(st["status"], "BLOCKED")

    def test_interrupted_run_discovery(self):
        out_nslkdd = get_artifact_directory(self.spec_nslkdd, base_dir=self.tmp_dir)
        write_status(out_nslkdd, status="RUNNING")

        discovered = self.scheduler.discover_run_states([self.spec_nslkdd])
        self.assertEqual(len(discovered["interrupted"]), 1)

    def test_provenance_safe_resume_reuse(self):
        out_nslkdd = get_artifact_directory(self.spec_nslkdd, base_dir=self.tmp_dir)
        write_status(out_nslkdd, status="COMPLETED")
        write_metrics(out_nslkdd, {
            "final_accuracy": 0.98,
            "final_f1_macro": 0.97,
            "final_attack_success_rate": 0.02,
        })
        # Write matching config.json
        from campaign.config_freezer import freeze_configuration
        freeze_configuration(self.spec_nslkdd, out_nslkdd, allow_dirty=True)

        discovered = self.scheduler.discover_run_states([self.spec_nslkdd])
        self.assertEqual(len(discovered["completed"]), 1)

        # Schedule should reuse cache without launching worker
        res = self.scheduler.schedule([self.spec_nslkdd], verbose=False)
        self.assertEqual(res["resumed_cached"], 1)
        self.assertEqual(res["completed_new"], 0)


if __name__ == "__main__":
    unittest.main()
