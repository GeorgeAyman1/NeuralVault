from datetime import datetime, UTC, timedelta

from core.memory.decay import MemoryDecay


def test_recent_memory_has_high_decay_score():
    decay = MemoryDecay()
    now = datetime.now(UTC)

    score = decay.score(
        created_at=now.isoformat(),
        metadata={"importance": 0.0},
    )

    assert score > 0.99


def test_old_memory_has_lower_decay_score():
    decay = MemoryDecay()
    old_time = datetime.now(UTC) - timedelta(days=90)

    score = decay.score(
        created_at=old_time.isoformat(),
        metadata={"importance": 0.0},
    )

    assert score < 0.1


def test_important_memory_decays_more_slowly():
    decay = MemoryDecay()
    old_time = datetime.now(UTC) - timedelta(days=90)

    low_importance_score = decay.score(
        created_at=old_time.isoformat(),
        metadata={"importance": 0.0},
    )

    high_importance_score = decay.score(
        created_at=old_time.isoformat(),
        metadata={"importance": 1.0},
    )

    assert high_importance_score > low_importance_score
