# Running experiments on the remote box (`sana`)

The eval box is an always-on Azure VM. It exists because the local pilot
(`2026-08-30-web-arm-subset20/`) lost 5 tasks to macOS sleep — `threading.Timer`
waits on a monotonic clock that stops while the laptop sleeps, so the timeout
never fired while wall-clock runtime kept counting. A remote box cannot sleep.

| | |
|---|---|
| host | `52.186.168.61` (hostname `sana-eval`, Azure, private IP `172.16.0.4`) |
| user | `asw2215` |
| key | `sana-eval_key.pem` in the repo root — **gitignored** (`.gitignore:24`, `*.pem`) |
| OS | Ubuntu 24.04.4 LTS, kernel `6.17.0-1022-azure` |
| specs | 8 vCPU, 31 GB RAM, 122 GB free on `/` |
| remote checkout | `~/sana-framework` (see `REMOTE_DIR` below) |

> Note: `2026-08-31-web-arm-subset20b/PLAN.md` §2 calls this "remote EC2". It is
> Azure, not EC2. Nothing depends on the difference — the scripts only need ssh,
> tar, and python3 — but the plan's wording is wrong.

## One-time setup on a new laptop

**1. Fix the key's permissions.** OpenSSH silently ignores a key that others can
read, and the failure surfaces as `Permission denied (publickey)`, which looks
like the key is wrong rather than merely unreadable:

```bash
chmod 600 sana-eval_key.pem
```

This resets whenever the key is re-downloaded or re-copied.

**2. Add an ssh alias** to `~/.ssh/config` so neither you nor any script needs
to carry the IP and `-i` around:

```
Host sana
    HostName 52.186.168.61
    User asw2215
    IdentityFile /Users/austinsenna/Documents/projects/daplab/sana-framework/sana-eval_key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 5
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 10m
```

```bash
mkdir -p ~/.ssh/cm    # ControlPath directory must exist
```

`ServerAlive*` keeps an idle session from being dropped by the NAT in front of
the VM. `ControlMaster` reuses one TCP connection for every subsequent `ssh`,
which is what makes the many small `ssh sana '...'` calls in the run scripts
fast instead of paying a full handshake each time.

## Connecting

```bash
ssh sana                                  # interactive shell
ssh sana 'uname -srm; df -h ~'            # one-shot command
ssh -t sana 'tmux attach -t sana'         # attach the shared tmux session
```

A long-lived tmux session named `sana` is kept on the box for shared work.
Recreate it if it is gone:

```bash
ssh sana 'tmux new-session -d -s sana -x 200 -y 50'
```

Anything started inside tmux survives a disconnect, a laptop sleep, and a closed
terminal — which is the entire point of running here. Nothing long-running
should be launched outside it.

### Driving tmux without a TTY

Agent tools and `!`-prefixed commands in Claude Code get no pseudo-terminal, so
`ssh sana` alone connects, prints the banner, and exits. Send keystrokes to the
session and read the screen back instead:

```bash
ssh sana "tmux send-keys -t sana 'tail -f logs/driver.log' Enter"
ssh sana 'tmux capture-pane -p -t sana | tail -40'
```

## Running a sweep

The `*_remote.sh` scripts in each experiment directory take `REMOTE_HOST` and an
optional `REMOTE_IDENTITY`. With the `sana` alias in place, `REMOTE_IDENTITY` is
unnecessary — the config already supplies the key:

```bash
cd experiments/2026-08-31-web-arm-subset20b
REMOTE_HOST=sana ./bootstrap_remote.sh    # ship code + .env + lance_data, build venv, preflight
REMOTE_HOST=sana ./run_remote.sh          # launch the four arms, detached in its own tmux session
REMOTE_HOST=sana ./pull_remote.sh         # tar results/ and logs/ back into this directory
REMOTE_HOST=sana ./watch_and_pull.sh      # poll, and pull after each arm finishes (instead of waiting for all four)
```

`run_remote.sh` creates its **own** tmux session per sweep (`webarm-<timestamp>`),
separate from the shared `sana` session, and prints the attach and tail commands
when it starts. Override `MODEL=` to change the model and `REMOTE_DIR=` to change
the remote checkout path.

`bootstrap_remote.sh` ships the tree with `git archive HEAD` — **committed state
only**. Uncommitted local edits do not reach the box. Two things are shipped
separately because they are gitignored: `.env` and `lance_data/` (~759 MB, and it
cannot be rebuilt remotely — the FTS index needs `datalake_with_schema.parquet`,
which is not on disk). `.agents/` is gitignored and is *not* shipped, so the
semantic judge is absent on the remote and `semantic_match` cannot be produced
there; bootstrap prints a warning about this at the end.
