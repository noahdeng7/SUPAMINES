import bisect
import os
import os.path
from filelock import FileLock
import numpy as np

from . import tetris

TAP_SEQUENCE_MAP = {
    '30hz': np.arange(10) * 2,
    '24hz': np.array([0, 3, 5, 8, 10, 13, 15, 18, 20, 23]),
    '20hz': np.arange(10) * 3,
    '15hz': np.arange(10) * 4,
    '12hz': np.arange(10) * 5,
    '10hz': np.arange(10) * 6,
    'slow5': np.array([0, 2, 4, 6, 18, 20, 22, 24, 36, 38]),
}
# Order fixes the tap axis of GameParamManager.param_count, so it is not free to change:
# a saved cnt*.npz is indexed by it.
TAP_NAMES = ('30hz', '24hz', '20hz', '15hz', '12hz', '10hz', 'slow5')
TAP_SEQUENCES = [TAP_SEQUENCE_MAP[i] for i in TAP_NAMES]
TAP_IDS = {name: i for i, name in enumerate(TAP_NAMES)}
ADJ_DELAYS = [0, 18, 21, 24, 30, 61]
ADJ_DELAY_IDS = {delay: i for i, delay in enumerate(ADJ_DELAYS)}
BUCKET_INTERVAL = 5
LINE_CAP = tetris.Tetris.LineCap()
BUCKETS = LINE_CAP // BUCKET_INTERVAL - 1
# Gravity changes at 130 / 230 / 330 lines (core/game.h: kLevelSpeedLines), which is what
# splits a game into the four regimes below.  The last entry is the build's line cap, so a
# variant built with a shorter cap simply has fewer of them.
LEVEL_LINES = [i for i in (0, 130, 230, 330) if i < LINE_CAP] + [LINE_CAP]

# --- The episode (NEW_DESIGN_SPEC.md section 1) -------------------------------------------
# One episode is a level-18 start played to one of exactly two terminals: TOP_OUT, or the
# MILESTONE at 230 lines (level-29 entry), which costs nothing.  230 is a *runtime* setting
# rather than a build constant so that a different milestone is a flag and not a rebuild;
# the C++ side takes it per env (PythonTetris::SetMilestone).
MILESTONE_LINES = min(230, LINE_CAP)

# --- Level segments ---------------------------------------------------------------------
# A *segment* is one of those gravity regimes, and a segment expert is a model trained on
# nothing else: the level-18 expert never sees 19+ gravity, the 29 expert never sees
# anything slower than 29.  The four are exactly the environment's own LevelSpeed classes,
# so a state's segment is already in its observation -- the one-hot in `move_meta[0:4]`,
# written by GetState -- and nothing has to carry a segment id next to an observation.
NUM_SEGMENTS = len(LEVEL_LINES) - 1
# Named by the level each one starts at, which is how these regimes are always referred to:
# "18", "19", "29", "39" -- covering levels 18, 19-28, 29-38 and 39-48 respectively.
SEGMENT_LEVELS = (18, 19, 29, 39)[:NUM_SEGMENTS]
SEGMENT_NAMES = tuple(str(i) for i in SEGMENT_LEVELS)
# The tap speed and reaction time the experts are trained at.  Everything else in the
# TAP_SEQUENCES x ADJ_DELAYS grid is binned rather than sampled; see GameParamManager.
EXPERT_TAP_ID = 0        # TAP_SEQUENCES[0] == 30hz
EXPERT_ADJ_DELAY_ID = 1  # ADJ_DELAYS[1] == 18 frames
# Boards from other segments to drop before giving up on the board file for one episode.
kMaxBoardSkips = 4096


def tap_ids_for(spec):
    """`[tap id]` for one tap-speed name, or None for 'all' -- i.e. no restriction."""
    if spec is None or spec == 'all': return None
    if spec not in TAP_IDS:
        raise ValueError('unknown tap speed {!r}; expected one of {}, or "all"'.format(
            spec, ', '.join(TAP_NAMES)))
    return [TAP_IDS[spec]]


def adj_delay_ids_for(spec):
    """`[adj delay id]` for one adjustment delay in frames, or None for 'all'."""
    if spec is None or spec == 'all': return None
    delay = int(spec)
    if delay not in ADJ_DELAY_IDS:
        raise ValueError('unknown adjustment delay {!r}; expected one of {}, or "all"'.format(
            spec, ', '.join(str(i) for i in ADJ_DELAYS)))
    return [ADJ_DELAY_IDS[delay]]


def level_by_lines(lines):
    """The NES level a line count is at (core/game.h: GetLevelByLines).

    Spec section 1 writes it as `18 if lines < 130 else 19 + (lines - 130) // 10`, which is
    the same function: 129 -> 18, 130 -> 19, 229 -> 28, 230 -> 29.
    """
    return 18 if lines < 130 else lines // 10 + 6


def segment_id(segment):
    """Normalize a segment given as an index (0..3) or as its starting level (29, '29')."""
    if isinstance(segment, str):
        if not segment.lstrip('-').isdigit():
            raise ValueError('unknown segment {!r}; expected one of {}'.format(
                segment, ', '.join(SEGMENT_NAMES)))
        segment = int(segment)
    if segment in SEGMENT_LEVELS: return SEGMENT_LEVELS.index(segment)
    if 0 <= segment < NUM_SEGMENTS: return int(segment)
    raise ValueError('unknown segment {!r}; expected one of {} (or an index 0..{})'.format(
        segment, ', '.join(SEGMENT_NAMES), NUM_SEGMENTS - 1))


def segment_or_none(segment):
    """`segment_id`, but None for the unrestricted case and every spelling of it.

    Config values arrive as strings (RL/config.py builds its flags by type), so "none" and
    "all" have to mean the same thing as leaving it out.
    """
    if segment is None: return None
    if isinstance(segment, str) and segment.strip().lower() in ('', 'none', 'all'):
        return None
    return segment_id(segment)


def segment_lines(segment):
    """[start, end) absolute line counts of a segment."""
    segment = segment_id(segment)
    return LEVEL_LINES[segment], LEVEL_LINES[segment + 1]


def segment_of_lines(lines):
    """The segment a line count falls in; anything past the cap belongs to the last one."""
    return min(bisect.bisect_right(LEVEL_LINES, lines) - 1, NUM_SEGMENTS - 1)


def segment_buckets(segment):
    """[start, end) indices into the bucket axis of GameParamManager.param_count."""
    start, end = segment_lines(segment)
    return start // BUCKET_INTERVAL, min(end // BUCKET_INTERVAL, BUCKETS)


def segment_label(segment):
    """'29 (levels 29-38, lines 230-329)' -- for logs and --help."""
    segment = segment_id(segment)
    start, end = segment_lines(segment)
    levels = '{}'.format(SEGMENT_LEVELS[segment])
    if level_by_lines(end - 1) != SEGMENT_LEVELS[segment]:
        levels = '{}-{}'.format(SEGMENT_LEVELS[segment], level_by_lines(end - 1))
    return '{} (level{} {}, lines {}-{})'.format(
        SEGMENT_NAMES[segment], '' if levels.isdigit() else 's', levels, start, end - 1)


class GameParamManager:
    """Per-episode parameters: what an episode starts from, and the input model it runs at.

    The episode the spec defines starts at level 18 with an empty board and a zeroed line
    counter, and that is what this returns by default -- `mid_ratio` and `board_ratio` are
    both 0, so every draw is a *true start*.

    Randomized mid-game resets (`mid_ratio`) and curriculum boards (`board_ratio`) are
    strong training aids and are allowed, but they change the state distribution and
    therefore what p_hat means.  So every draw is stamped with `is_true_start`, and the dual
    variable and every reported number are measured on those episodes alone (spec section
    9).  Turning either ratio up without reserving evaluation envs -- BatchedGames'
    `eval_ratio` -- leaves `c` constraining a distribution nobody plays.

    `segment`, `tap_ids` and `adj_delay_ids` restrict what may be sampled at all.  Anything
    outside them is *binned* -- given infinite count, so the sampler never returns it --
    rather than merely down-weighted.  A pinned segment other than the first starts every
    episode inside its own line range, so none of its episodes is a true start.
    """

    def __init__(self, board_file, segment=None, tap_ids=None, adj_delay_ids=None,
                 milestone=MILESTONE_LINES):
        self.eof = False
        self.board_file = board_file
        self.segment = segment_or_none(segment)
        self.milestone = int(milestone) if milestone else LINE_CAP
        # A pinned segment cuts the episode at its end -- a truncation, not a terminal.  A
        # segment that runs into the milestone needs no cut: the milestone ends it first.
        self.line_cap = 0 if self.segment is None else LEVEL_LINES[self.segment + 1]
        if self.line_cap >= self.milestone: self.line_cap = 0
        self.tap_ids = None if tap_ids is None else sorted({int(i) for i in tap_ids})
        self.adj_delay_ids = None if adj_delay_ids is None else sorted({int(i) for i in adj_delay_ids})
        self.total_cnt = 0
        self.board_cnt = 0
        self.board_short_cnt = 0
        self.param_count = np.zeros((len(TAP_SEQUENCES), len(ADJ_DELAYS), BUCKETS), dtype='int64')
        self.mid_ratio = 0.0
        self.board_ratio = 0.0
        self.short_ratio = 0.0
        # Binned combinations, masked out of every draw.
        self.excluded = np.zeros(self.param_count.shape, dtype='bool')
        if self.tap_ids is not None:
            self.excluded[[i for i in range(len(TAP_SEQUENCES)) if i not in self.tap_ids]] = True
        if self.adj_delay_ids is not None:
            self.excluded[:, [i for i in range(len(ADJ_DELAYS)) if i not in self.adj_delay_ids]] = True
        self.rng = np.random.default_rng()
        self._NextBatch()

    @property
    def true_start_lines(self):
        """Line count a true start begins at, or None if this run cannot produce one."""
        if self.segment is None: return 0
        start, _ = segment_lines(self.segment)
        return 0 if start == 0 else None

    def SaveParams(self):
        np.savez_compressed(f'cnt{os.getpid()}.npz', self.param_count)

    def UpdateParams(self, params):
        """(mid_ratio, board_ratio, short_ratio); a negative value also clears the counts."""
        if any([i < 0 for i in params]):
            self.param_count = np.zeros(self.param_count.shape, dtype='int64')
        self.total_cnt, self.board_cnt, self.board_short_cnt = 0, 0, 0
        self.mid_ratio, self.board_ratio, self.short_ratio = [abs(i) for i in params]

    def UpdateState(self, params, pieces: int, lines: int):
        self.total_cnt += pieces
        if params.get('is_board'):
            self.board_cnt += pieces
            if params['is_short']:
                self.board_short_cnt += pieces

        start_bucket = min(params['lines'] // BUCKET_INTERVAL, BUCKETS)
        end_bucket = min((params['lines'] + lines) // BUCKET_INTERVAL + 1, BUCKETS)
        self.param_count[params['tap_id'], params['adj_delay_id'], start_bucket:end_bucket] += 1

    def GetNewParam(self, force_true_start: bool = False):
        """One episode's reset parameters.

        `force_true_start` is what BatchedGames' evaluation envs pass: those play the spec's
        episode and nothing else, whatever the training distribution is doing.
        """
        if force_true_start and self.true_start_lines is not None:
            return self._SampleDistribution(bucket_start=0, bucket_end=1, lines=0)
        skipped = 0
        while self.board_ratio > 0:
            data = self._GetNewBoard()
            if data is None: break
            # A curriculum board is filed under the segment it was sampled at, so an expert
            # takes only its own and drops the rest.  A board file may hold none for this
            # segment at all, and nothing else ends the loop -- the board budget is only
            # spent by boards that are actually used -- so bound the search and fall through
            # to an ordinary sampled episode.
            if self.segment is not None and data[2] != self.segment:
                skipped += 1
                if skipped >= kMaxBoardSkips: break
                continue
            bucket_start = LEVEL_LINES[data[2]] // BUCKET_INTERVAL
            bucket_end = LEVEL_LINES[data[2] + 1] // BUCKET_INTERVAL
            params = self._SampleDistribution(data[4], bucket_start, bucket_end)
            params['is_true_start'] = False
            return {'now_piece': data[1], 'is_board': True, 'board': data[0],
                    'is_short': data[3], **params}
        # A mid-game start is drawn only `mid_ratio` of the time; the rest -- and all of it
        # by default -- is the level-18 start the spec defines.
        if self.true_start_lines is not None and self.rng.random() >= self.mid_ratio:
            return self._SampleDistribution(bucket_start=0, bucket_end=1, lines=0)
        return self._SampleDistribution()

    def _SampleDistribution(self, cells: int = 0, bucket_start: int = 0,
                            bucket_end: int = BUCKETS, lines=None):
        n_count = self.param_count.astype('float32')
        if self.segment is not None:
            seg_start, seg_end = segment_buckets(self.segment)
            bucket_start, bucket_end = max(bucket_start, seg_start), min(bucket_end, seg_end)
        # Nothing may start at or past the milestone: that episode would be over before it
        # began.  So the bucket axis stops one bucket short of it.
        bucket_start = min(bucket_start, BUCKETS)
        bucket_end = min(bucket_end, BUCKETS, max(1, self.milestone // BUCKET_INTERVAL))
        if bucket_start >= bucket_end:
            raise RuntimeError('no line bucket left to sample: segment {}, milestone {}'.format(
                self.segment, self.milestone))
        n_count[self.excluded] = np.inf
        if bucket_start > 0: n_count[:,:,:bucket_start] = np.inf
        if bucket_end < BUCKETS: n_count[:,:,bucket_end:] = np.inf
        if not np.isfinite(n_count).any():
            raise RuntimeError(
                'nothing left to sample: segment {}, tap ids {}, adj delay ids {}'.format(
                    self.segment, self.tap_ids, self.adj_delay_ids))
        # Sampling from softmax(-n_count) is the Gumbel-max trick: argmax(-n_count + G) with
        # G = -log(-log(U)), i.e. argmin(n_count + log(-log(U))).  That avoids normalizing and
        # the cumsum/searchsorted inside rng.choice(p=...); doing it in float32 in place makes
        # it ~4x cheaper again than rng.gumbel(), and this runs once per finished episode.
        flat = n_count.reshape(-1)
        flat -= flat.min() # keep float32 precision in the comparison below
        g = self.rng.random(flat.size, dtype='float32')
        with np.errstate(divide='ignore'): # a uniform of exactly 0 just means "never pick this"
            np.log(g, out=g)
            np.negative(g, out=g)
            np.log(g, out=g)
        c = np.argmin(flat + g)
        tap_id, adj_delay_id, bucket = np.unravel_index(c, n_count.shape)
        if lines is None:
            start_lines = bucket * BUCKET_INTERVAL
            # Tetris::Reset derives the piece count as (lines * 10 + cells) / 4 and refuses
            # a combination that does not divide.
            if (start_lines % 2 != 0) != (cells % 4 != 0): start_lines += 1
            lines = start_lines + 2 * self.rng.integers(
                ((bucket + 1) * BUCKET_INTERVAL - start_lines + 1) // 2)
        return {'tap_sequence': TAP_SEQUENCES[tap_id].tolist(), 'tap_id': tap_id,
                'adj_delay': ADJ_DELAYS[adj_delay_id], 'adj_delay_id': adj_delay_id,
                'lines': int(lines), 'line_cap': self.line_cap, 'milestone': self.milestone,
                'is_true_start': int(lines) == 0}

    def _GetNewBoard(self):
        if self.eof: return None
        is_board = self.board_cnt < self.board_ratio * self.total_cnt
        if not is_board: return None

        if self.data_offset >= len(self.data):
            self._NextBatch()
            if self.data_offset >= len(self.data):
                return None
        is_short = self.board_short_cnt < self.short_ratio * self.board_cnt
        b, piece, level = self.data[self.data_offset]
        cells = 200 - int.from_bytes(b, 'little').bit_count()
        self.data_offset += 1
        return tetris.Board(b), piece, level, is_short, cells

    def _NextBatch(self):
        if not self.board_file:
            self.eof = True
            return
        self.data = self.read_board_file(self.board_file)
        self.data_offset = 0

    @staticmethod
    def read_board_file(board_file: str, chunk_size: int = 4096):
        DATA_SIZE = 26
        board_file_offset_f = board_file + '.offset'
        board_file_lock = board_file + '.lock'
        lock = FileLock(board_file_lock)
        with lock:
            if os.path.isfile(board_file_offset_f):
                with open(board_file_offset_f, 'r') as f: offset = int(f.read().strip())
            else:
                offset = 0
            with open(board_file, 'rb') as f:
                f.seek(offset * DATA_SIZE)
                data = f.read(chunk_size * DATA_SIZE)
                ret = [(data[i*DATA_SIZE:i*DATA_SIZE+DATA_SIZE-1],
                        data[i*DATA_SIZE+DATA_SIZE-1]&7,
                        data[i*DATA_SIZE+DATA_SIZE-1]>>3) for i in range(len(data) // DATA_SIZE)]
            with open(board_file_offset_f, 'w') as f:
                print(0 if len(ret) == 0 else offset + len(ret), file=f)
        return ret
