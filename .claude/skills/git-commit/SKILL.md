---
name: git-commit
description: Create git commits with a single-line Conventional Commits message and no AI attribution. Use this skill whenever the user asks to commit, "commit the changes", "commit this", "save this to git", "make a commit", stage and commit, or asks for a commit message — even if they don't mention the word "skill" or "conventional commits".
---

# Git Commit

Commit staged/unstaged work with a short, clean, conventional commit message.

## Hard rules

1. **No attribution to Claude, ever.** Do not add `Co-Authored-By: Claude`, `Generated with Claude Code`, `🤖`, or any trailer, footer, or line mentioning Claude, Anthropic, or AI. The commit author and committer stay as the user's configured git identity. Never pass `--author`.
2. **One line. That's it.** The message is a single Conventional Commits subject line. No body, no bullet list, no "why" paragraph, no footers. If the change feels too big for one line, that's a signal to split it into multiple commits, not to write a longer message.
3. **Never use `--no-verify`.** If hooks fail, fix the problem or report it.
4. **Never push unless explicitly asked.** Commit only.

## Message format

```
<type>(<optional scope>): <description>
```

- **Types:** `feat`, `fix`, `refactor`, `perf`, `docs`, `style`, `test`, `build`, `ci`, `chore`, `revert`
- **Scope:** optional, lowercase, the module/area touched — `auth`, `api`, `ui`, `deps`
- **Description:** imperative mood ("add", not "added"/"adds"), lowercase first letter, no trailing period
- **Length:** aim for ≤ 60 chars, hard limit 72
- **Breaking changes:** `feat(api)!: drop v1 endpoints`

Good:
```
feat(auth): add refresh token rotation
fix(parser): handle empty input without panic
refactor: extract retry logic into helper
chore(deps): bump axios to 1.7.9
docs: document env vars in readme
```

Bad:
```
Fixed the bug.                              # past tense, vague, period
feat: add stuff                             # meaningless
update files                                # no type
feat(auth): add refresh token rotation

This change introduces...                   # body not wanted
```

## Workflow

1. Run in parallel: `git status`, `git diff HEAD`, and `git log --oneline -10`.
   - The log is for **matching the repo's existing style** — if the repo doesn't use Conventional Commits, follow the repo's convention instead of imposing this one.
2. If a commit template or `commitlint`/`.gitmessage`/`CONTRIBUTING.md` convention exists, that wins over the defaults here.
3. Stage what belongs in this commit. Prefer explicit paths over `git add -A` so nothing unintended sneaks in. Never stage `.env`, credentials, keys, tokens, large binaries, or local-only config — flag those instead.
4. Write the message from the **actual diff**, describing the net effect, not the process.
5. Commit with a single `-m`:
   ```bash
   git commit -m "feat(auth): add refresh token rotation"
   ```
6. Run `git status` after, to confirm it landed and see what's still uncommitted.

## Multiple logical changes

If the working tree mixes unrelated changes, split them into separate commits rather than writing one vague message. Stage per-file (or with `git add -p` when a single file holds unrelated hunks) and commit each group. Briefly tell the user how you split it.

## Failure handling

- **Pre-commit hook modified files:** re-stage the affected files and retry the commit once. If it fails again, stop and report.
- **Pre-commit hook rejected the commit:** report the failure and what needs fixing. Don't bypass it.
- **Nothing staged and nothing to stage:** say so; don't create an empty commit.
- **Amending:** only amend when explicitly asked, and never amend a commit that's already been pushed.
