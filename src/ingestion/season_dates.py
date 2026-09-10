"""Season-name -> wide UTC unix date bracket.

`/fixtures/between` requires `from`/`to` unix timestamps in addition to
the (empirically-confirmed-working but undocumented) `seasons` filter.
We pass a bracket wide enough to guarantee every fixture in the season
falls inside it (July 1 of the start year through August 15 of the end
year, which comfortably covers preseason-to-postseason including any
schedule slippage) and let the `competitions`+`seasons` filters do the
precise selection.

Known API behavior -- 365-day range warning (observed, not a bug):
the `from`/`to` bracket produced here spans more than 365 days (it
covers a full season plus buffer), and OddAlerts' response includes
`info.warning: "Date range exceeds 365 days. Results have been limited
to 365 days from the start date."` on every page. This was confirmed
during the Phase 1.1 EPL 2024/25 smoke test (both raw response pages
carried this warning) and does NOT truncate the actual results: the
`seasons` filter parameter independently constrains which fixtures are
returned regardless of the from/to window's width, and the smoke test
returned exactly the expected 380 fixtures for that season with no
missing matches (cross-checked fixture-ID-for-fixture-ID against the
normalized DB -- see docs/PHASE1_1_SECURITY_SCHEMA_REMEDIATION.md).
The date-window logic is therefore left as-is; this docstring exists
so a future maintainer doesn't "fix" the warning by narrowing the
bracket without first re-confirming there's no actual data loss to fix.
"""
from __future__ import annotations

from datetime import datetime, timezone


def season_unix_bracket(season_name: str) -> tuple[int, int]:
    """'2023/2024' -> (unix for 2023-07-01T00:00:00Z, unix for 2024-08-15T00:00:00Z)."""
    start_year_str, end_year_str = season_name.split("/")
    start_year, end_year = int(start_year_str), int(end_year_str)
    start = datetime(start_year, 7, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 8, 15, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())
