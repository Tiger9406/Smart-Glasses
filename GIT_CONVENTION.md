# Git convention

Issues, commits, and pull requests should be easy to scan. The title says what kind of work it is. The description says what it is and how you know it is done.

Project board: https://github.com/users/Tiger9406/projects/2/views/1

Add the ticket to that board when you open it. Tickets from the hardware repo still have to be added by hand until the repos are linked on GitHub Pro.

## Issues

Start the title with one of these:

| Tag | Use it when |
| --- | --- |
| `[Spike]` | You need to figure something out before anyone can build it. A prototype, a comparison, a "does this even work." The result is an answer, not the finished change. |
| `[Bug]` | Something is broken or behaving wrong. |
| `[Task]` | A defined chunk of work that is not a new capability and not a defect. Refactors, wiring, cleanup, chores. |
| `[Feature]` | New behavior someone can actually use. |
| `[Epic]` | The work is too big for one ticket. The epic is the outcome. The real work lives in sub-issues under it. |

`[Spike]`, `[Bug]`, `[Task]`, and `[Feature]` are normal tickets. If you cannot describe the finish line in one sitting, it is an `[Epic]`. Open the epic, then open sub-issues for the pieces and put the same tags on those.

And just because we combined all into one repo, let's have `[Software]` `[Hardware]` `[Firmware]`
as well.

```
[Task] Type the worker queues
```

```
[Epic] Replace the identity flow
```

Description:

```
## What
- [what this is, and enough context that someone else knows why it exists]

## DoD
- [the checks that mean this issue is done]
```

`DoD` is the definition of done. If every line is checked, the ticket can close.

## Commits

Lead with the issue number, then a short summary of the change.

```
[44] Fix api plugging
```

One space after the bracket. The number is the GitHub issue this commit belongs to. There should be like max 3 commits per pull request—I actually learn to only have one commit per pr but I feel like it's extreme. Main message is if your pr gets bigger than 3 commits it's probably too big a pr to read for other people. Use discretion for how big to make each commit — I think of each commit as around 3-4 hrs of work.

## Pull requests

Title matches the commit: issue number, then the same kind of summary.

```
[44] Fix api plugging
```

Description:

```
## What
[what you changed, why, and the same context you put on the ticket so the reviewer does not have to hunt for it]

Resolves https://github.com/Tiger9406/Smart-Glasses/issues/44

## Tests you wrote
- [how you checked this, including tests you ran and anything you tried by hand]
```

Use the real issue link. `Resolves` closes that issue when the pull request merges.
