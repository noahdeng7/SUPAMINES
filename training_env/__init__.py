"""The NES Tetris environment used for rollouts.

`tetris` is the compiled extension (board representation, move search, frame
sequencing, batched stepping); `game`/`game_param` are the thin Python layer that
drives it and samples per-episode parameters.
"""

from . import tetris
from .game import BatchedGames
from .game_param import (
    ADJ_DELAYS,
    ADJ_DELAY_IDS,
    BUCKETS,
    BUCKET_INTERVAL,
    EXPERT_ADJ_DELAY_ID,
    EXPERT_TAP_ID,
    GameParamManager,
    LEVEL_LINES,
    LINE_CAP,
    MILESTONE_LINES,
    NUM_SEGMENTS,
    SEGMENT_LEVELS,
    SEGMENT_NAMES,
    TAP_IDS,
    TAP_NAMES,
    TAP_SEQUENCES,
    TAP_SEQUENCE_MAP,
    adj_delay_ids_for,
    level_by_lines,
    segment_buckets,
    segment_id,
    segment_label,
    segment_lines,
    segment_of_lines,
    segment_or_none,
    tap_ids_for,
)

__all__ = [
    'tetris', 'BatchedGames', 'GameParamManager',
    'TAP_SEQUENCE_MAP', 'TAP_SEQUENCES', 'TAP_NAMES', 'TAP_IDS',
    'ADJ_DELAYS', 'ADJ_DELAY_IDS',
    'BUCKET_INTERVAL', 'BUCKETS', 'LEVEL_LINES', 'LINE_CAP', 'MILESTONE_LINES',
    # level segments -- one expert per segment; see game_param.py
    'NUM_SEGMENTS', 'SEGMENT_LEVELS', 'SEGMENT_NAMES',
    'EXPERT_TAP_ID', 'EXPERT_ADJ_DELAY_ID',
    'segment_id', 'segment_or_none', 'segment_lines', 'segment_of_lines', 'segment_buckets',
    'segment_label',
    'level_by_lines', 'tap_ids_for', 'adj_delay_ids_for',
]
