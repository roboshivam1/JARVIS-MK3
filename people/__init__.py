"""People database — who Shivam knows, what's true about them, what's owed.

Flow: debrief text → extract.py (LLM) → proposals → review.py (approve/reject)
      → real rows in the people / facts / commitments / key_dates tables.
"""
from .store import PeopleStore
from .review import ReviewQueue, ReviewError
from .extract import DebriefExtractor

__all__ = ["PeopleStore", "ReviewQueue", "ReviewError", "DebriefExtractor"]
