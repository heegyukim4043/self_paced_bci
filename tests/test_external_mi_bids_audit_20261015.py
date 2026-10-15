"""Fail-closed checks for the external MI raw-data inventory."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_external_mi_bids_20261015.py"
SPEC = importlib.util.spec_from_file_location("audit_external_mi_bids", SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class ExternalMiBidsAuditTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        eeg_dir = root / "sub-1" / "ses-0" / "eeg"
        eeg_dir.mkdir(parents=True)
        base = "sub-1_ses-0_task-imagery_run-0"
        (root / "participants.tsv").write_text("participant_id\nsub-1\n", encoding="utf-8")
        (eeg_dir / f"{base}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n"
            "1\t4\tleft_hand\n9\t4\tright_hand\n", encoding="utf-8")
        (eeg_dir / f"{base}_eeg.json").write_text(
            '{"SamplingFrequency":1000,"EEGChannelCount":2}', encoding="utf-8")
        (eeg_dir / f"{base}_channels.tsv").write_text(
            "name\ttype\tunits\nC3\tEEG\tuV\nC4\tEEG\tuV\n", encoding="utf-8")
        (eeg_dir / f"{base}_eeg.set").write_bytes(b"EEG" * 1024)
        return eeg_dir / f"{base}_events.tsv"

    def test_inventory_never_promotes_unlabelled_gaps_to_idle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._fixture(root)
            report = audit_module.audit(root, "wbcic")
            self.assertEqual(report["n_subjects_in_events"], 1)
            self.assertEqual(report["run_count_by_subject_session"], {"1|0": 1})
            self.assertEqual(report["subjects_missing_from_participants"], [])
            self.assertEqual(report["n_runs_without_eeg_file"], 0)
            self.assertEqual(report["event_types_all_runs"],
                             {"left_hand": 1, "right_hand": 1})
            self.assertEqual(report["runs"][0]["eeg_sidecar"]["SamplingFrequency"], 1000)
            self.assertFalse(report["training_ready"])
            self.assertFalse(any("idle_seconds" in key for key in report))
            self.assertIn("unlabelled gaps are not idle", " ".join(report["unresolved_gates"]))

    def test_absent_events_cannot_look_like_a_completed_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "no .*events.tsv"):
                audit_module.audit(Path(temp), "dreyer")

    def test_partial_scope_missing_eeg_and_invalid_event_are_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self._fixture(root)
            path.write_text("onset\tduration\ttrial_type\n9\t4\tleft\n1\tbad\tright\n",
                            encoding="utf-8")
            path.with_name(path.name.replace("_events.tsv", "_eeg.set")).unlink()
            report = audit_module.audit(root, "dreyer", max_events_files=1)
            self.assertEqual(report["scope"], "partial")
            self.assertEqual(report["n_runs_without_eeg_file"], 1)
            self.assertIn("events are not chronological in file order",
                          report["runs"][0]["events"]["issues"])
            self.assertIn("line 3: invalid duration", report["runs"][0]["events"]["issues"])
            self.assertFalse(report["training_ready"])


if __name__ == "__main__":
    unittest.main()
