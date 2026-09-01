# demo

Plays live games in TetrisGYM running under FCEUX.

```
fceux.py            model server -- listens for the Lua script and returns placements
lua/TetrisGYMv5.lua FCEUX script -- reads NES memory, sends state, replays inputs
culour.py           terminal colour helper for the curses board display
example_threshold   sample threshold table (TETRIS_ONLY builds)
```

## Running

```sh
python demo/fceux.py models/distilled_checkpoint.pt
python demo/fceux.py models/<checkpoint>.pth       # or a single checkpoint
```

Then open FCEUX, load TetrisGYM, and run `lua/TetrisGYMv5.lua`. The script's
configuration block at the top of the file sets `server_url` / `server_port` (default
`127.0.0.1:3456`, matching `fceux.py`), `start_level`, whether to set seeds explicitly,
and whether to play one game or a batch.

If `setseed` is false, navigate to the level-selection screen *before* starting the
script; otherwise it navigates there itself.

`-c/--use-curses` renders the board in the terminal as it plays.

> On Windows, `import curses` needs `pip install windows-curses`; it is not in the
> stdlib there, and `fceux.py` imports it unconditionally regardless of `--use-curses`.

## How it works

The Lua script polls NES memory for game state (piece, position, level, lines, board)
and sends it over a TCP socket. `fceux.py` mirrors that state into a `training_env`
`Tetris` instance, runs the policy to pick a placement, and then calls the environment's
`GetSequence()` to turn that placement into the frame-level button sequence that reaches
it. It is that sequence, not the placement, that goes back over the socket for the Lua
side to replay. Because the sequence comes from the environment's move search, it
respects the configured tap speed and adjustment delay — the agent is constrained to
inputs a human controller could actually produce.

`--adj-delay` and `--tap-speed` must match the regime the checkpoint was trained under.

`--milestone` defaults to 0 here, meaning "no early milestone": live play runs a whole game
to the build line cap rather than stopping at the 230-line milestone an RL episode ends at.

## Flags that referenced the removed tablebase

`-s/--server` and `--tablebase-cutoff` connect to a tablebase board server to override
the policy's choice near the end of a game, where exact play beats a learned policy.
**That server was part of the tablebase code removed from this repo** (`src/server.cpp`,
recoverable from git history at `f68460d`). The client code here still works and is left
intact, but there is nothing in-tree to point it at. Leave both unset unless you are
running that server from elsewhere.
