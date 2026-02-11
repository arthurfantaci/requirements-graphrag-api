# PR Workflow

Guided pull request creation and merge following professional git workflow best practices.

Use when the user says `/pr-workflow`, "create a PR", "open a pull request", or "merge a PR".

**Educational mandate**: Always explain the "why" behind each step. The user is building
professional git workflow expertise — never skip explanations even for routine operations.

## Determine workflow type

Ask: "Is this a simple PR (one branch, one issue) or part of a stacked set?"

---

## Simple PR (one branch → one issue)

### 1. Identify the associated issue(s)

```bash
git log --oneline origin/main..HEAD
gh issue list --state open --limit 10
```

Ask: "Which GitHub issue(s) does this PR close?"

### 2. Gather PR context

Run in parallel:
```bash
git status
git diff origin/main --stat
git log --oneline origin/main..HEAD
```

### 3. Create the PR

**Title:** `<type>: brief description` (under 70 chars)
Types: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`

```bash
gh pr create --title "<type>: brief description" --body "$(cat <<'EOF'
Closes #<issue_number>

## Summary
- Bullet point changes (2-4 bullets)

## Test plan
- [ ] Tests pass (`uv run pytest`)
- [ ] Lint clean (`uv run ruff check`)
- [ ] Build passes (if frontend changes: `npm run build`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 4. Post-creation verify

```bash
gh pr view --json title,body,url
```

### 5. Merge — choose the right strategy

Recommend based on situation and explain the tradeoff:

| Strategy | Command | When to use | Tradeoff |
|---|---|---|---|
| **Squash** | `gh pr merge <n> --squash` | Most simple PRs — clean 1-commit history | Loses individual commit detail |
| **Merge commit** | `gh pr merge <n> --merge` | When commit-by-commit history matters, or preserving author attribution across multiple commits | Adds a merge commit to history |
| **Rebase** | `gh pr merge <n> --rebase` | Linear history preference, small PRs with clean commits | Rewrites commit SHAs |

**Default recommendation for this project**: `--squash` for simple PRs (keeps `git log` clean).

### 6. Post-merge verification (ALWAYS do this)

```bash
# Verify the issue auto-closed
gh issue view <N> --json state --jq .state

# If NOT closed (shows "OPEN"), close manually
gh issue close <N> --comment "Delivered in PR #<pr_number>"

# Clean up local branch
git checkout main && git pull
git branch -D <branch-name>
```

Explain: "GitHub only auto-closes issues when the PR body contains `Closes #N`.
Let's verify it worked."

---

## Stacked PRs (multiple dependent branches)

Use when work spans multiple issues and each phase depends on the previous.

### When to stack vs. use one big PR

- **Stack** when: phases need separate review, or you want incremental CI validation
- **One big PR** when: solo project, all changes are tightly coupled, simpler workflow
- **Rule of thumb**: avoid stacks deeper than 3 — the rebase tax compounds

### 1. Create branches in sequence

```bash
git checkout main && git pull
git checkout -b phase-1
# ... work, commit ...
git checkout -b phase-2   # branches from phase-1
# ... work, commit ...
```

### 2. Create PRs — each with its own Closes keyword

Each PR body MUST close its own issue(s):

```bash
# PR for phase 1
gh pr create --title "refactor: phase 1 work" --body "$(cat <<'EOF'
Closes #10

## Summary
...
EOF
)"

# PR for phase 2 (targets phase-1 branch, NOT main)
gh pr create --base phase-1 --title "refactor: phase 2 work" --body "$(cat <<'EOF'
Closes #11

## Summary
...
EOF
)"
```

### 3. Choose merge strategy for the stack

**This is the critical decision.** Explain the tradeoff clearly before proceeding:

#### Option A: Squash merge (clean history, manual rebase between each)

Best when: you want one clean commit per phase in the final history.

After squash-merging PR #1, PR #2's branch has orphaned parent commits.
You must rebase manually after each merge:

```bash
# After PR #1 is squash-merged:
git fetch origin main

# Recreate phase-2 branch from updated main + only its own commit
git checkout -B phase-2 origin/main
git cherry-pick <phase-2-specific-commit-sha>
git push --force-with-lease origin phase-2

# Retarget PR to main (if it was targeting phase-1 branch)
gh pr edit <pr-number> --base main

# Wait for CI, then merge
```

Repeat for each subsequent phase.

#### Option B: Merge commits (preserves branch topology, no rebase needed)

Best when: stack is deep (>3), or you want to avoid the rebase tax.

```bash
# Merge each PR sequentially with merge commits
gh pr merge <n> --merge
# GitHub auto-retargets the next PR to main — no manual rebase needed
```

Explain: "Merge commits preserve the parent-commit linkage, so downstream branches
remain compatible. The tradeoff is a noisier git log with merge commits."

#### Option C: Collapse the stack into one PR

Best when: solo project, or the phases are tightly coupled and separate review isn't needed.

```bash
# Close intermediate PRs
gh pr close <intermediate-pr> --comment "Collapsing into single PR #<final>"

# The final PR's branch already has all cumulative changes
# Add ALL Closes keywords to its body
gh pr edit <final-pr> --body "$(cat <<'EOF'
Closes #10, Closes #11, Closes #12

## Summary
...
EOF
)"
gh pr merge <final-pr> --squash
```

### 4. Post-merge verification (ALWAYS do this for every PR in the stack)

```bash
# Check ALL issues that should have closed
for issue in 10 11 12; do
  echo "#$issue: $(gh issue view $issue --json state --jq .state)"
done

# Close any that didn't auto-close
gh issue close <N> --comment "Delivered in PR #X"

# Close any orphaned PRs from superseded branches
gh pr close <N> --comment "Superseded by PR #X"

# Clean up local branches
git checkout main && git pull
git branch -D phase-1 phase-2 phase-3
```

### 5. If earlier PRs get superseded

If you merge a later PR that contains all earlier work (common when stacks collapse):
- The earlier PRs won't auto-close their issues (different commit SHAs after squash)
- Close orphaned issues manually: `gh issue close <N> --comment "Delivered in PR #X"`
- Close orphaned PRs: `gh pr close <N> --comment "Superseded by PR #X"`

---

## Closing keyword rules (both workflows)

**CRITICAL** — these auto-close issues when the PR merges:
- `Closes #N`, `Fixes #N`, or `Resolves #N`
- Multiple: `Closes #10, Closes #11`
- Must be in the PR **body** (not title, not comments)

**What does NOT work:**
- `(#42)` in title — just a mention/hyperlink
- Closing keyword in a PR comment added later
- Closing keyword in commit messages on non-default branches

No associated issue? Write "No associated issue" in the body (bypasses the hook).

---

## CI/CD awareness

Before merging, always verify CI status:
```bash
gh pr view <n> --json statusCheckRollup --jq '[.statusCheckRollup[] | {name, status, conclusion}]'
```

Explain what each check does:
- **Lint & Format Check**: Runs `ruff check` and `ruff format --check`
- **Test**: Runs `pytest` with the full test suite
- **Vercel**: Preview deployment of the frontend
- **Evaluation** (on tags/schedule only): Tier 3 benchmark on release, Tier 4 nightly

If CI fails, diagnose before merging. Never skip failing checks.

## Common mistakes

1. **`(#42)` in title** — mention, not close command
2. **Squash-merge + stacked PR** without rebasing downstream branches
3. **Forgetting `--base`** on stacked PRs (defaults to main, shows all cumulative changes)
4. **Not verifying** issue state after merge — always check
5. **Choosing merge strategy without explaining tradeoff** — always tell the user why
6. **Merging with failing CI** — always check status first
