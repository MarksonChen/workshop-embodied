"""Immutable animal/session splits for Demo B motion-model comparisons.

Coltrane is the default workshop animal because its original 281-D model is the
behaviorally validated locomotion checkpoint.  ``DEMO_B_ANIMAL=freddie`` is an
explicit rejected comparison, not the workshop default.
"""

from __future__ import annotations

import os


FREDDIE_TRAIN_SESSIONS = (
    "2022_05_16_1", "2022_05_17_1", "2022_05_20_1", "2022_05_23_1",
    "2022_05_24_1", "2022_05_25_1", "2022_05_27_1", "2022_05_30_1",
    "2022_05_31_1", "2022_06_01_1", "2022_06_03_1", "2022_06_08_1",
    "2022_06_09_1", "2022_06_12_1", "2022_06_15_1", "2022_06_17_1",
    "2022_06_20_1",
)
FREDDIE_VAL_SESSIONS = (
    "2022_05_19_1", "2022_05_26_1", "2022_06_02_1", "2022_06_13_1",
)
FREDDIE_TEST_SESSIONS = (
    "2022_05_21_1", "2022_05_28_1", "2022_06_06_1", "2022_06_16_1",
)

# Validation and test dates are interleaved across Coltrane's recording
# calendar.  The remaining 26 sessions form train; no crop crosses a file.
COLTRANE_VAL_SESSIONS = (
    "2021_07_30_1", "2021_08_05_1", "2021_08_11_1",
    "2021_08_17_1", "2021_08_23_2", "2021_09_04_1",
)
COLTRANE_TEST_SESSIONS = (
    "2021_08_01_1", "2021_08_07_1", "2021_08_13_1",
    "2021_08_19_1", "2021_08_27_1", "2021_09_15_1",
)
COLTRANE_TRAIN_SESSIONS = (
    "2021_07_28_1", "2021_07_29_1", "2021_07_31_1", "2021_08_02_1",
    "2021_08_03_1", "2021_08_04_1", "2021_08_06_1", "2021_08_08_1",
    "2021_08_09_1", "2021_08_10_1", "2021_08_12_1", "2021_08_14_1",
    "2021_08_15_1", "2021_08_16_1", "2021_08_18_1", "2021_08_20_1",
    "2021_08_21_1", "2021_08_23_1", "2021_08_25_1", "2021_08_30_1",
    "2021_08_31_1", "2021_09_03_1", "2021_09_14_1", "2021_09_16_1",
    "2021_09_17_1", "2021_09_18_1",
)

# The validated standalone transition was trained on these eight chronological
# sessions.  Demo E treats every other session as genuinely held out rather
# than reusing the broader split above, which was designed for fresh model
# training and overlaps the standalone model's historical corpus.
COLTRANE_PRIOR_TRAIN_SESSIONS = (
    "2021_07_28_1", "2021_07_29_1", "2021_07_30_1", "2021_07_31_1",
    "2021_08_01_1", "2021_08_02_1", "2021_08_03_1", "2021_08_04_1",
)
COLTRANE_PRIOR_VAL_SESSIONS = (
    "2021_08_05_1", "2021_08_11_1", "2021_08_17_1",
    "2021_08_23_2", "2021_09_04_1",
)
COLTRANE_PRIOR_TEST_SESSIONS = (
    "2021_08_07_1", "2021_08_13_1", "2021_08_19_1",
    "2021_08_27_1", "2021_09_15_1",
)


def validate_coltrane_prior_split() -> None:
    groups = (
        set(COLTRANE_PRIOR_TRAIN_SESSIONS),
        set(COLTRANE_PRIOR_VAL_SESSIONS),
        set(COLTRANE_PRIOR_TEST_SESSIONS),
    )
    assert tuple(map(len, groups)) == (8, 5, 5)
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])

SPLITS = {
    "freddie": (
        FREDDIE_TRAIN_SESSIONS,
        FREDDIE_VAL_SESSIONS,
        FREDDIE_TEST_SESSIONS,
    ),
    "coltrane": (
        COLTRANE_TRAIN_SESSIONS,
        COLTRANE_VAL_SESSIONS,
        COLTRANE_TEST_SESSIONS,
    ),
}


def split_for(animal: str):
    try:
        return SPLITS[animal.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported Demo B animal {animal!r}; choose {sorted(SPLITS)}") from error


ANIMAL = os.environ.get("DEMO_B_ANIMAL", "coltrane").lower()
TRAIN_SESSIONS, VAL_SESSIONS, TEST_SESSIONS = split_for(ANIMAL)
ALL_SESSIONS = TRAIN_SESSIONS + VAL_SESSIONS + TEST_SESSIONS


def validate_split() -> None:
    expected = {"freddie": (17, 4, 4), "coltrane": (26, 6, 6)}[ANIMAL]
    assert (len(TRAIN_SESSIONS), len(VAL_SESSIONS), len(TEST_SESSIONS)) == expected
    assert not (set(TRAIN_SESSIONS) & set(VAL_SESSIONS))
    assert not (set(TRAIN_SESSIONS) & set(TEST_SESSIONS))
    assert not (set(VAL_SESSIONS) & set(TEST_SESSIONS))
    assert len(set(ALL_SESSIONS)) == sum(expected)
