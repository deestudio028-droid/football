import pytest
from datetime import datetime, timezone
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import FixtureService

def test_get_weekly_prediction_fixtures_offsets():
    fs = FixtureService()
    
    # Current week
    fixtures_curr, meta_curr = fs.get_weekly_prediction_fixtures(offset_weeks=0)
    assert len(fixtures_curr) == 48
    
    # Upcoming week
    fixtures_upcoming, meta_upcoming = fs.get_weekly_prediction_fixtures(offset_weeks=1)
    # The count might be less than 48 if the feed doesn't have enough, but it shouldn't be the same exact fixtures
    curr_ids = {f.fixture_id for f in fixtures_curr}
    upcoming_ids = {f.fixture_id for f in fixtures_upcoming}
    
    # Assert that there's minimal overlap (there shouldn't be any overlap unless the league has no upcoming games)
    overlap = curr_ids.intersection(upcoming_ids)
    assert len(overlap) < 48  # Should be totally disjoint or mostly disjoint
    
    print(f"Current week count: {len(fixtures_curr)}")
    print(f"Upcoming week count: {len(fixtures_upcoming)}")
    print(f"Overlap: {len(overlap)}")
