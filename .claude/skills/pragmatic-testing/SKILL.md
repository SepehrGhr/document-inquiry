---
name: pragmatic-testing
description: Guidelines for writing and running tests in a sensible, proportionate way. Use this skill whenever the user asks to write tests, add test coverage, test a function or feature, verify that code works, or whenever you are about to add tests after implementing something. Also use it when deciding how much testing is enough. Keeps tests focused on real behavior and prevents over-testing, absurd edge cases, and bloated test suites.
---

# Pragmatic Testing

The goal of testing is to answer one question: **"Does this code do what it is supposed to do for the people and systems that actually use it?"**

Write the tests a thoughtful senior developer would write in a normal code review. Not the tests a paranoid auditor would write. A small, readable test suite that catches real bugs beats a huge one that tests everything.

## Core principles

1. **Test behavior, not implementation.** Check what a function returns or what a feature does, not how it does it internally. Tests should survive a refactor.
2. **Test what is realistic.** Only test inputs and situations that can actually happen in this project. If you can't explain how a case would occur in real use, skip it.
3. **Proportion effort to risk.** Core logic, money, data loss, security, and anything that has broken before deserve more care. Simple getters, config, and glue code deserve little or none.
4. **Less is more.** Each test must earn its place by catching a bug that matters. If removing a test would not reduce your confidence, don't write it.

## What to test (in this order)

1. **The happy path.** The normal, expected use. Does it work when everything is fine?
2. **The main variations.** The few different normal ways the function or feature gets used (e.g. different valid input types, optional parameters on/off).
3. **Realistic edge cases.** Only the ones that plausibly occur:
   - empty input (empty list, empty string) when users or callers could really send it
   - boundaries that matter to the logic (0, 1, max allowed, "just over the limit")
   - missing or null values where the code actually receives them
4. **Realistic failures.** What should happen when something expected goes wrong: invalid user input, a failed network request, a missing file, a denied permission. Check that the error is handled sensibly.
5. **Bugs you just fixed.** Add one regression test per real bug.

## What NOT to test

- Impossible states: inputs the type system, validation layer, or calling code already prevents.
- Third-party libraries, the language, or the framework. Assume they work.
- Trivial code: simple getters/setters, constants, pass-through wrappers, plain data classes.
- Every combination of every parameter. Pick representative cases instead.
- Absurd inputs: gigantic payloads, exotic unicode, race conditions, clock tricks, or memory exhaustion, unless this project clearly deals with them (e.g. a text processor handling international text, or a payment system).
- Private/internal helpers on their own. Cover them through the public function that uses them.
- The same behavior twice at different levels. One good test is enough.
- Exact wording of log messages, error text, or styling, unless users depend on it.

## Sizing guide

Use these as rough ceilings, not targets. Fewer is fine.

| Thing being tested | Typical number of tests |
|---|---|
| Simple function | 2–4 |
| Function with real branching logic | 4–8 |
| A feature or endpoint | 1–2 end-to-end happy paths + a few key failures |
| Bug fix | 1 regression test |

If you find yourself writing more than ~8 tests for one function, stop and ask: are these actually different behaviors, or the same one in disguise?

## Before writing each test, ask

- Could this really happen in this app? If no, skip it.
- Would a real user or caller notice if this broke? If no, skip it.
- Does this test overlap with one I already have? If yes, skip it.
- Will this test break when I change the implementation but not the behavior? If yes, rewrite it.

## How to work

1. **Read the code first** and understand what it is meant to do and who calls it.
2. **Check existing tests** and follow the project's current style, framework, and folder layout. Don't introduce a new test framework or heavy tooling unless asked.
3. **Write the few most valuable tests**, starting with the happy path.
4. **Run them.** Make sure they pass, and that they would fail if the code were broken (don't write tests that can never fail).
5. **Stop.** Don't keep adding tests to chase coverage numbers or because you can think of more cases.

## Keep tests simple

- One clear idea per test, with a name that says what it checks (`returns_empty_list_when_no_items`).
- Minimal setup. Avoid elaborate mocks; mock only things that are slow, external, or non-deterministic (network, database, time).
- Plain, readable assertions. No clever loops or logic inside tests.
- Don't touch real external services, user data, or production resources from tests.

## When in doubt

If you're unsure whether a case is worth testing, **leave it out and mention it in one line** at the end (e.g. "I didn't test X because it can't happen with the current validation"). Let the user decide whether they want it. Never pile on extra tests "just to be safe."

## Final summary

When you finish, briefly report:
- what you tested and why those cases matter
- what you deliberately skipped
- the test results
