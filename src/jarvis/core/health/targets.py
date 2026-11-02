# =============================================================================
# src/jarvis/core/health/targets.py - what a good day looks like, in numbers
# =============================================================================
#
# Targets to reach and limits to stay under, from public guidelines and
# the owner's profile. Any of them can be overridden by name in the
# profile ("target_protein_g" = "130"), and an override always wins.
#
#   protein      1.6 g per kg body weight - the evidence-based floor for
#                building muscle while lifting (ranges run 1.6-2.2).
#   calories     Mifflin-St Jeor resting energy x activity, plus a small
#                surplus (+250 kcal) for lean muscle gain. Needs weight.
#   fibre        30 g a day (ICMR-NIN, roughly 15 g per 1,000 kcal).
#   sodium       under 2,000 mg a day (WHO: under 5 g of salt).
#   added sugar  under 50 g a day (WHO: under 10% of energy).
#   junk         fried or ultra-processed items: under 4 a week.
#
# These are general-population numbers, not medical advice. Nothing here
# goes below a sensible floor: there is no restriction mode.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

ACTIVITY = {"0": 1.2, "1": 1.3, "2": 1.375, "3": 1.45, "4": 1.5, "5": 1.55, "6": 1.6, "7": 1.7}
SURPLUS = {"gain_muscle": 250, "maintain": 0, "lose_fat": -300}


@dataclass(frozen=True)
class Targets:
    kcal: float | None            # a target to reach (needs weight)
    protein_g: float | None       # a target to reach (needs weight)
    fibre_g: float                # a target to reach
    sodium_mg: float              # a limit
    sugar_g: float                # a limit
    junk_per_week: int            # a limit

    def as_dict(self) -> dict[str, float | int | None]:
        return {"kcal": self.kcal, "protein_g": self.protein_g, "fibre_g": self.fibre_g,
                "sodium_mg": self.sodium_mg, "sugar_g": self.sugar_g,
                "junk_per_week": self.junk_per_week}


def _num(profile: dict[str, str], key: str) -> float | None:
    try:
        value = float(profile[key])
    except (KeyError, ValueError):
        return None
    return value if value > 0 else None


def compute(profile: dict[str, str]) -> Targets:
    weight = _num(profile, "weight_kg")
    height = _num(profile, "height_cm")
    age = _num(profile, "age")
    goal = profile.get("goal", "maintain")

    protein = round(weight * 1.6) if weight else None
    kcal = None
    if weight and height and age:
        sex_term = 5 if profile.get("sex", "male") == "male" else -161
        bmr = 10 * weight + 6.25 * height - 5 * age + sex_term
        factor = ACTIVITY.get(profile.get("training_days", "4"), 1.5)
        floor = 1600 if profile.get("sex", "male") == "male" else 1300
        kcal = max(floor, round((bmr * factor + SURPLUS.get(goal, 0)) / 50) * 50)

    def pick(name: str, default: float | None) -> float | None:
        override = _num(profile, f"target_{name}")
        return override if override is not None else default

    return Targets(
        kcal=pick("kcal", kcal),
        protein_g=pick("protein_g", protein),
        fibre_g=pick("fibre_g", 30) or 30,
        sodium_mg=pick("sodium_mg", 2000) or 2000,
        sugar_g=pick("sugar_g", 50) or 50,
        junk_per_week=int(pick("junk_per_week", 4) or 4),
    )
