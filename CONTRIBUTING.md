# Contributing

Bug reports and patches are welcome.

## Before you open a PR

```bash
python -m compileall azizgpt scripts     # must pass
python scripts/test_tools.py --self-test # scorer sanity, no API calls
ruff check azizgpt scripts               # if you have ruff
```

The full harness (`python scripts/test_tools.py`) makes real API calls and
consumes roughly a third of the Groq free daily token budget, so run it
deliberately rather than habitually, and paste the `SCORE:` line in the PR.

## Ground rules for tool code

The security model is the reason this project exists, so changes under
`azizgpt/tools/` are held to it:

- No `shell=True`, no `eval`, no `exec`, ever.
- Every subprocess takes an argv list. Model output never becomes part of a
  command.
- New tools need a JSON schema **and** independent argument validation inside
  the function. The schema constrains the model; it does not enforce anything.
- Anything destructive or irreversible needs a confirmation gate. If you add
  affirmative words to `power.AFFIRMATIVE`, check first that Whisper does not
  hallucinate them from silence — that is why "ok", "okay" and "sure" are
  excluded.
- Never log or print a key. Report a bad key by shape, not by value.

## Style

Match the surrounding code. Comments explain *why*, not *what*; the codebase
favours a short note about the reasoning over a restatement of the line below.
