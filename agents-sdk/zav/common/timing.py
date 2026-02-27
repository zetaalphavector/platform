from datetime import datetime, timezone

from pydantic import AwareDatetime

TimezoneAwareDatetime = AwareDatetime


def now() -> datetime:
    return datetime.now(timezone.utc)
