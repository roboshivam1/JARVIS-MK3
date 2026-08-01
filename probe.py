import asyncio
from pathlib import Path

from jarvis.common.facts import Fact
from jarvis.core.db.database import Database
from jarvis.core.db.repos.facts import FactsRepo, _fts_query


async def main() -> None:
    path = Path("/tmp/fts_probe.db")
    path.unlink(missing_ok=True)          # fresh every run

    db = await Database.connect(path)
    await db.migrate()
    repo = FactsRepo(db)

    await repo.store(Fact(text="Shivam is building SqOnion, a wall-mounted purifier"))
    await repo.store(Fact(text="Shivam studies engineering at LNMIIT"))

    for query in ["SqOnion", "studying", "purifier"]:
        built = _fts_query(query)
        print(f"\nquery {query!r}  ->  FTS: {built!r}")

        rows = await db.query(
            "SELECT bm25(facts_fts) AS rank, text FROM facts_fts "
            "WHERE facts_fts MATCH ?",
            (built,),
        )
        print("  raw rows:", [(round(r["rank"], 4), r["text"][:35]) for r in rows])
        print("  repo hits:", [
            (round(score, 4), fact.text[:35])
            for fact, score in await repo.search_keyword(query)
        ])

    await db.close()


asyncio.run(main())