import _pathfix  # noqa: F401
import json
import logging
import tempfile
import unittest
from pathlib import Path

from ingestion.config import ConfigError, load_config
from ingestion.logging_setup import close_all_logging, redact, setup_logging


def _write_project(tmp_path: Path, token: str = "SECRET_TOKEN_VALUE") -> Path:
    (tmp_path / ".env").write_text(f"OddAlerts_API = {token}\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "competitions.json").write_text(json.dumps({
        "competitions": [
            {"name": "Premier League", "competition_id": 423, "country_id": 45,
             "seasons": [{"season_id": 6484, "season_name": "2024/2025"}]}
        ]
    }), encoding="utf-8")
    return tmp_path


class TestConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_config_reads_token_from_dotenv(self):
        _write_project(self.tmp_path)
        config = load_config(self.tmp_path)
        self.assertEqual(config.api_token, "SECRET_TOKEN_VALUE")
        self.assertEqual(config.base_url, "https://data.oddalerts.com/api")
        self.assertEqual(len(config.competitions), 1)
        self.assertEqual(config.competitions[0].competition_id, 423)

    def test_load_config_missing_env_raises(self):
        (self.tmp_path / "config").mkdir()
        (self.tmp_path / "config" / "competitions.json").write_text(
            '{"competitions": []}', encoding="utf-8"
        )
        with self.assertRaises(ConfigError):
            load_config(self.tmp_path)

    def test_config_repr_never_exposes_token(self):
        _write_project(self.tmp_path)
        config = load_config(self.tmp_path)
        text = repr(config)
        self.assertNotIn("SECRET_TOKEN_VALUE", text)
        self.assertIn("redacted", text)


class TestLogging(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        # Release every log-file handle this test caused to be opened
        # BEFORE deleting the temporary directory holding those files.
        # logging.getLogger(name) is a process-wide singleton, so a
        # logger configured during a test stays alive (with its
        # FileHandler attached and its file open) after the test method
        # returns. POSIX allows unlinking an open file, so this was
        # invisible on Linux; Windows refuses with WinError 32.
        # This is deterministic resource cleanup, NOT suppression --
        # no cleanup error is ignored and no assertion is relaxed.
        close_all_logging()
        self._tmp.cleanup()

    def test_redact_scrubs_token_from_arbitrary_text(self):
        text = "GET https://data.oddalerts.com/api/fixtures/between?api_token=SECRET_TOKEN_VALUE&page=2"
        self.assertNotIn("SECRET_TOKEN_VALUE", redact(text))
        self.assertIn("api_token=<redacted>", redact(text))

    def test_logger_file_handler_never_writes_token(self):
        logger = setup_logging(self.tmp_path / "logs", name="test_logger")
        logger.warning("Request failed for url with api_token=SECRET_TOKEN_VALUE embedded")
        for handler in logger.handlers:
            handler.flush()
        log_contents = (self.tmp_path / "logs" / "test_logger.log").read_text(encoding="utf-8")
        self.assertNotIn("SECRET_TOKEN_VALUE", log_contents)
        self.assertIn("<redacted>", log_contents)

    def test_calling_setup_logging_again_closes_the_previous_file_handler(self):
        # Fix 3 (WinError 32): logging.getLogger(name) is a process-wide
        # singleton -- a second setup_logging() call with the same name
        # must close() the first FileHandler before detaching it, not
        # just drop the reference and leak the OS file handle. This is
        # what let a stale handle from a previous call keep a tempdir's
        # log file open on Windows, causing tempdir cleanup to fail with
        # WinError 32.
        logger = setup_logging(self.tmp_path / "logs_a", name="test_logger_reopen")
        first_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(first_file_handlers), 1)
        first_handler = first_file_handlers[0]
        self.assertIsNotNone(first_handler.stream, "first handler's stream should be open right after setup")

        setup_logging(self.tmp_path / "logs_b", name="test_logger_reopen")

        # The first handler's underlying stream must be closed (closing a
        # logging.FileHandler sets .stream to None) -- proving the file
        # descriptor was actually released, not just forgotten about.
        self.assertIsNone(first_handler.stream, "previous FileHandler's stream must be closed, not just detached")

    def test_setup_logging_never_raises_when_called_repeatedly_with_same_name(self):
        # A regression guard for the underlying WinError 32 symptom
        # itself: repeated setup_logging() calls (as would happen across
        # multiple tests/runs in the same process) must never raise.
        for _ in range(5):
            setup_logging(self.tmp_path / "logs_repeat", name="test_logger_repeat")

    def test_close_all_logging_releases_every_file_handle(self):
        # The regression this guards: a logger configured but never
        # re-configured kept its FileHandler open past the end of the
        # test, locking the file on Windows. close_all_logging() must
        # release it deterministically.
        setup_logging(self.tmp_path / "logs_close_a", name="test_logger_close_a")
        setup_logging(self.tmp_path / "logs_close_b", name="test_logger_close_b")

        close_all_logging()

        for name in ("test_logger_close_a", "test_logger_close_b"):
            logger = logging.getLogger(name)
            self.assertEqual(logger.handlers, [], f"{name} still has handlers attached")

    def test_log_file_is_deletable_after_close_all_logging_without_reopen(self):
        # Directly reproduces the Windows failure mode: configure once,
        # write, then delete the file with no second setup_logging() call.
        log_dir = self.tmp_path / "logs_single"
        logger = setup_logging(log_dir, name="test_logger_single")
        logger.warning("something happened")

        close_all_logging()

        log_file = log_dir / "test_logger_single.log"
        self.assertTrue(log_file.exists())
        log_file.unlink()  # would raise WinError 32 on Windows if the handle leaked
        self.assertFalse(log_file.exists())

    def test_close_all_logging_is_idempotent(self):
        setup_logging(self.tmp_path / "logs_idem", name="test_logger_idem")
        close_all_logging()
        close_all_logging()  # must not raise on an already-released logger

    def test_logging_still_works_after_close_and_reconfigure(self):
        # close_all_logging() must not disable logging -- a later
        # setup_logging() call reconfigures normally, redaction included.
        setup_logging(self.tmp_path / "logs_r1", name="test_logger_reconfig")
        close_all_logging()

        logger = setup_logging(self.tmp_path / "logs_r2", name="test_logger_reconfig")
        logger.warning("url with api_token=SECRET_TOKEN_VALUE inside")
        for handler in logger.handlers:
            handler.flush()

        contents = (self.tmp_path / "logs_r2" / "test_logger_reconfig.log").read_text(encoding="utf-8")
        self.assertNotIn("SECRET_TOKEN_VALUE", contents)
        self.assertIn("<redacted>", contents)

    def test_repeated_setup_does_not_accumulate_duplicate_handlers(self):
        for _ in range(4):
            logger = setup_logging(self.tmp_path / "logs_dup", name="test_logger_dup")
        # Exactly one FileHandler + one StreamHandler, never 4 of each.
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)
        self.assertEqual(len(stream_handlers), 1)

    def test_old_log_file_is_deletable_immediately_after_reopen(self):
        # The most direct simulation of the Windows failure mode: after
        # setup_logging() is called a second time with the same name,
        # the FIRST run's log file must be deletable/replaceable while
        # the process is still running -- this is exactly what tempdir
        # cleanup needs to succeed on Windows.
        first_log_dir = self.tmp_path / "logs_c"
        setup_logging(first_log_dir, name="test_logger_delete")
        setup_logging(self.tmp_path / "logs_d", name="test_logger_delete")

        log_file = first_log_dir / "test_logger_delete.log"
        self.assertTrue(log_file.exists())
        log_file.unlink()  # must not raise (would be WinError 32 on Windows if the handle leaked)
        self.assertFalse(log_file.exists())


if __name__ == "__main__":
    unittest.main()
