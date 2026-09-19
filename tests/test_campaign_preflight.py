"""
tests/test_campaign_preflight.py
Unit tests for campaign preflight integrity checks.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from campaign.preflight import CampaignPreflight, PreflightError
from campaign.runner import check_dataset_availability


class TestCampaignPreflight(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.preflight = CampaignPreflight(
            base_results_dir=self.tmp_dir,
            allow_dirty=True,
            min_disk_gb=1.0,
            min_ram_gb=0.2,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_preflight_all_checks_pass_with_allow_dirty(self):
        res = self.preflight.run_all_checks(verbose=False)
        self.assertTrue(res["all_passed"])
        self.assertEqual(res["canonical_runs"], 6111)
        self.assertIn("1_clean_git_tree", res["checks"])
        self.assertIn("4_canonical_count", res["checks"])
        self.assertIn("5_zero_duplicate_ids", res["checks"])

    def test_preflight_fails_on_dirty_tree(self):
        strict_preflight = CampaignPreflight(
            base_results_dir=self.tmp_dir,
            allow_dirty=False,
            min_disk_gb=1.0,
            min_ram_gb=0.2,
        )
        with patch("campaign.preflight.git_state") as mock_git:
            mock_git.return_value = {
                "git_commit": "a" * 40,
                "git_dirty": True,
                "git_dirty_paths": ["modified_file.py"],
            }
            with self.assertRaises(PreflightError) as ctx:
                strict_preflight.run_all_checks(verbose=False)
            self.assertIn("Clean Git tree failed", str(ctx.exception))

    def test_preflight_fails_on_invalid_commit(self):
        with patch("campaign.preflight.git_state") as mock_git:
            mock_git.return_value = {
                "git_commit": "short_sha",
                "git_dirty": False,
                "git_dirty_paths": [],
            }
            with self.assertRaises(PreflightError) as ctx:
                self.preflight.run_all_checks(verbose=False)
            self.assertIn("Valid commit failed", str(ctx.exception))

    def test_preflight_fails_on_insufficient_disk(self):
        with patch("shutil.disk_usage") as mock_disk:
            # Return 500 MB free (0.5 GB), requirement is 5.0 GB
            mock_disk.return_value = (100 * 1024**3, 99.5 * 1024**3, 500 * 1024**2)
            strict_disk_preflight = CampaignPreflight(
                base_results_dir=self.tmp_dir,
                allow_dirty=True,
                min_disk_gb=5.0,
            )
            with self.assertRaises(PreflightError) as ctx:
                strict_disk_preflight.run_all_checks(verbose=False)
            self.assertIn("Insufficient disk space", str(ctx.exception))

    def test_preflight_fails_on_insufficient_ram(self):
        with patch("campaign.preflight.get_system_ram_gb") as mock_ram:
            mock_ram.return_value = (0.05, 1.0)  # 50 MB available, 1 GB total
            strict_ram_preflight = CampaignPreflight(
                base_results_dir=self.tmp_dir,
                allow_dirty=True,
                min_ram_gb=1.0,
            )
            with self.assertRaises(PreflightError) as ctx:
                strict_ram_preflight.run_all_checks(verbose=False)
            self.assertIn("Insufficient compute capacity", str(ctx.exception))

    def test_dataset_availability_gating(self):
        nslkdd_ok, _ = check_dataset_availability("nslkdd")
        self.assertTrue(nslkdd_ok)

        ciciot_ok, ciciot_reason = check_dataset_availability("ciciot2023")
        self.assertFalse(ciciot_ok)
        self.assertIn("BLOCKED", ciciot_reason)

        edge_ok, edge_reason = check_dataset_availability("edgeiiotset")
        self.assertFalse(edge_ok)
        self.assertIn("BLOCKED", edge_reason)


if __name__ == "__main__":
    unittest.main()
