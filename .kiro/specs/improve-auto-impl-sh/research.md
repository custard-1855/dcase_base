# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `improve-auto-impl-sh`
- **Discovery Scope**: New Feature (greenfield script development)
- **Key Findings**:
  - Bash best practices require `set -euo pipefail` for defensive scripting
  - `select` built-in command provides standard interactive menu functionality
  - Configuration precedence: CLI args > environment vars > config file > defaults
  - ShellCheck integration is standard for static analysis

## Research Log

### Bash Error Handling and Retry Mechanisms

- **Context**: Requirements 2 and 4 require robust error handling with retry logic and validation step execution
- **Sources Consulted**:
  - [Red Hat Bash Error Handling](https://www.redhat.com/en/blog/bash-error-handling)
  - [Baeldung Retry Failed Command](https://www.baeldung.com/linux/shell-retry-failed-command)
  - [Stack Overflow Retry Logic](https://stackoverflow.com/questions/7449772/how-to-retry-a-command-in-bash)
- **Findings**:
  - **set -euo pipefail**: Defensive scripting pattern (errexit, nounset, pipefail) is standard
  - **Retry patterns**: Function-based retry with maximum attempts and sleep intervals
  - **trap command**: Essential for cleanup and error handling
  - **Exit status codes**: All commands return 0 for success, non-zero for failure
- **Implications**:
  - Script must use `set -euo pipefail` at the top for early failure detection
  - Retry logic should be implemented as reusable function with configurable max attempts
  - Error handling should use `trap` for cleanup on EXIT, ERR signals

### Interactive Menu Selection

- **Context**: Requirement 3 requires spec selection with interactive prompts
- **Sources Consulted**:
  - [Linuxize Bash Select](https://linuxize.com/post/bash-select/)
  - [Baeldung Simple Select Menu](https://www.baeldung.com/linux/shell-script-simple-select-menu)
  - [Ask Ubuntu Select Menu](https://askubuntu.com/questions/1705/how-can-i-create-a-select-menu-in-a-shell-script)
- **Findings**:
  - **select builtin**: Native bash construct for numbered menus
  - **PS3 variable**: Controls the prompt string
  - **$REPLY variable**: Stores the user's numeric selection
  - **$opt variable**: Stores the selected menu item text
  - **case statement**: Used with select for action dispatch
- **Implications**:
  - Use `select` loop for spec selection (no external dependencies)
  - Display spec metadata (phase, ready_for_implementation, pending task count) in menu
  - Combine with `jq` for parsing `.kiro/specs/*/spec.json`

### Logging with Timestamps and Rotation

- **Context**: Requirement 6 requires timestamped log files with automatic cleanup
- **Sources Consulted**:
  - [Better Stack Logrotate Guide](https://betterstack.com/community/guides/logging/how-to-manage-log-files-with-logrotate-on-ubuntu-20-04/)
  - [Dean Wilding Log Rotation Script](https://deanwilding.co.uk/2020/01/01/bash-script-to-rotate-a-log-file-to-a-time-stamped-file-linux/)
  - [Stack Overflow Log Rotation](https://stackoverflow.com/questions/3690015/bash-script-log-file-rotation)
- **Findings**:
  - **Timestamp format**: `date +%Y%m%d_%H%M%S` for sortable filenames
  - **logrotate**: Standard utility for production log management (overkill for this use case)
  - **Custom rotation**: Simple `find` command with `-mtime +N -delete` for age-based cleanup
  - **Redirection**: Use `tee` for simultaneous stdout and log file output
- **Implications**:
  - Generate log files as `.kiro/logs/auto_impl_YYYYMMDD_HHMMSS.log`
  - Use `tee -a` to append to log while showing output
  - Implement cleanup with `find .kiro/logs/ -name "auto_impl_*.log" -mtime +30 -delete`

### Configuration Management and Precedence

- **Context**: Requirement 5 requires flexible configuration with clear precedence
- **Sources Consulted**:
  - [Stack Overflow Config Precedence](https://stackoverflow.com/questions/11077223/what-order-of-reading-configuration-values)
  - [Unix SE Config Files](https://unix.stackexchange.com/questions/175648/use-config-file-for-my-shell-script)
- **Findings**:
  - **Standard precedence**: CLI args > environment vars > config file > defaults
  - **POSIX pattern**: Configuration → environment → command line
  - **Config file sourcing**: Use `. <config_file>` to source bash variable assignments
  - **Environment variable naming**: Convention is `KIRO_AUTO_IMPL_*` for namespacing
- **Implications**:
  - Load defaults first, then source `.kiro/auto_impl.config` if exists
  - Override with `KIRO_AUTO_IMPL_*` environment variables
  - Final override with command-line arguments (getopts or manual parsing)
  - Config file format: Simple `KEY=value` bash syntax

### ShellCheck and Defensive Scripting

- **Context**: Code quality and safety requirements (implicit from project tech stack)
- **Sources Consulted**:
  - [MIT Safe Shell Scripts](https://sipb.mit.edu/doc/safe-shell/)
  - [Bash Best Practices Cheat Sheet](https://bertvv.github.io/cheat-sheets/Bash.html)
  - [NameHero set -e -o pipefail Guide](https://www.namehero.com/blog/how-to-use-set-e-o-pipefail-in-bash-and-why/)
- **Findings**:
  - **set -e**: Exit on any command failure
  - **set -u**: Exit on unbound variable access
  - **set -o pipefail**: Fail pipeline if any command fails (not just last)
  - **ShellCheck**: Static analyzer for common bash mistakes (now POSIX 2024 standard)
  - **Quote expansion**: Always quote variable expansions `"${var}"`
- **Implications**:
  - Script header must include `set -euo pipefail`
  - All variable references must be quoted
  - Use ShellCheck during development (SC2086, SC2068 compliance)
  - Disable pipefail temporarily for commands where failure is expected

### Existing Project Patterns

- **Context**: Understanding current project bash script conventions
- **Sources Consulted**:
  - `DESED_task/dcase2024_task4_baseline/run_exp.sh`
  - `DESED_task/dcase2024_task4_baseline/run_exp_cmt.sh`
- **Findings**:
  - **uv run**: Standard command runner for Python scripts
  - **Timestamp generation**: `TIMESTAMP=$(date +"%Y%m%d_%H%M%S")`
  - **Log directory**: `mkdir -p logs/` pattern
  - **Output redirection**: `2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log`
  - **No error handling**: Current scripts lack `set -e`, `trap`, or retry logic
- **Implications**:
  - Follow existing timestamp and logging patterns for consistency
  - Improve on existing scripts by adding error handling and defensive programming
  - Use `uv run` prefix for Claude Code `/impl` command invocation

### Kiro Spec Structure

- **Context**: Understanding `.kiro/specs/` directory structure for spec selection
- **Sources Consulted**:
  - `.kiro/specs/sebbs-refactoring/tasks.md`
  - `.kiro/specs/refactor-sed-trainer/tasks.md`
  - `.kiro/specs/umap-visualization/tasks.md`
- **Findings**:
  - **Directory pattern**: `.kiro/specs/<feature-name>/`
  - **Required files**: `spec.json`, `requirements.md`, `design.md`, `tasks.md`
  - **Archives exclusion**: `archives/` directory should be ignored in spec listing
  - **spec.json fields**: `phase`, `ready_for_implementation`, `approvals.tasks.approved`
- **Implications**:
  - Spec discovery: `find .kiro/specs -mindepth 1 -maxdepth 1 -type d ! -name archives`
  - Parse `spec.json` with `jq` for metadata display
  - Validate `ready_for_implementation: true` before execution
  - Count pending tasks by parsing `tasks.md` (grep for `- [ ]` pattern)

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Monolithic Script | Single bash file with functions organized by responsibility | Simple deployment, no dependencies, easy to understand | Can grow large, harder to test individual components | **Selected** - Appropriate for automation script, follows existing project pattern |
| Modular Scripts | Separate scripts for spec selection, task execution, validation | Clearer separation of concerns, easier unit testing | Requires sourcing mechanism, more files to manage | Overkill for this use case |
| External Tool (Python) | Rewrite as Python script with subprocess calls | Better error handling, easier to test, richer libraries | Adds dependency, breaks from project's bash convention | Not aligned with existing `.sh` scripts in project |

## Design Decisions

### Decision: Use Native Bash with Defensive Programming

- **Context**: Need robust automation script for Kiro workflow
- **Alternatives Considered**:
  1. **Pure bash** - Minimal dependencies, follows project conventions
  2. **Python script** - Better libraries, easier testing
  3. **Bash with external tools** - Hybrid approach (dialog, fzf)
- **Selected Approach**: Pure bash with defensive programming patterns
- **Rationale**:
  - Consistency with existing `run_exp.sh` scripts in project
  - No additional dependencies beyond standard Unix tools (jq, find, grep)
  - Bash is sufficient for file system operations and command orchestration
  - Defensive patterns (set -euo pipefail, trap, retry functions) mitigate bash's weaknesses
- **Trade-offs**:
  - **Benefits**: Simple deployment, no runtime dependencies, matches project style
  - **Compromises**: Less robust testing framework compared to Python, more verbose error handling
- **Follow-up**: Validate error handling with integration tests (manual execution scenarios)

### Decision: select Built-in for Interactive Menu

- **Context**: Requirement 3 requires spec selection interface
- **Alternatives Considered**:
  1. **select built-in** - Native bash, no dependencies
  2. **dialog/whiptail** - Rich TUI, better UX
  3. **fzf** - Fuzzy finding, modern UX
- **Selected Approach**: Bash `select` built-in command
- **Rationale**:
  - Zero external dependencies (dialog/whiptail not always installed)
  - Standard bash feature since bash 2.x
  - Sufficient for numbered menu with 5-10 items (typical spec count)
  - Consistent with requirement's "interactive prompt" terminology
- **Trade-offs**:
  - **Benefits**: Portable, simple, well-documented
  - **Compromises**: No fuzzy search, basic visual presentation
- **Follow-up**: Consider fzf upgrade if spec count exceeds 20 or user feedback requests it

### Decision: Configuration File Format (KEY=value)

- **Context**: Requirement 5 requires flexible configuration loading
- **Alternatives Considered**:
  1. **Bash source format** (`KEY=value`) - Native, simple
  2. **YAML/TOML** - Structured, typed
  3. **INI format** - Sectioned, human-readable
- **Selected Approach**: Simple `KEY=value` bash source format
- **Rationale**:
  - Can be sourced directly with `. <file>` (no parsing needed)
  - Consistent with bash environment variable syntax
  - Supports comments (`# comment`)
  - No external parser dependencies
- **Trade-offs**:
  - **Benefits**: Zero parsing overhead, native bash semantics
  - **Compromises**: No type safety, no nested structures (flat namespace)
- **Follow-up**: Document config file syntax in script header comments

### Decision: Log Retention with find -mtime

- **Context**: Requirement 6.7 requires automatic old log deletion
- **Alternatives Considered**:
  1. **logrotate** - Standard utility, production-grade
  2. **find -mtime** - Simple, no config files
  3. **Manual cleanup** - User responsibility
- **Selected Approach**: `find .kiro/logs -name "auto_impl_*.log" -mtime +30 -delete`
- **Rationale**:
  - logrotate requires system-level configuration (not portable)
  - find is POSIX standard, available everywhere
  - Single command, no additional files or setup
  - Default 30 days aligns with development workflow (monthly cleanups)
- **Trade-offs**:
  - **Benefits**: Simple, portable, configurable via environment variable
  - **Compromises**: No compression, no rotation counts (only age-based)
- **Follow-up**: Make retention days configurable (`KIRO_AUTO_IMPL_LOG_RETENTION_DAYS`)

### Decision: Retry Mechanism with Exponential Backoff for API Limits

- **Context**: Requirement 4.6 requires API rate limit detection and automatic retry
- **Alternatives Considered**:
  1. **Fixed sleep** - Simple, predictable
  2. **Exponential backoff** - Standard for API retries
  3. **No retry** - Fail fast, user responsibility
- **Selected Approach**: Exponential backoff with max attempts (3 retries, 5s initial delay)
- **Rationale**:
  - API rate limits are transient errors (429 status codes)
  - Exponential backoff reduces server load and increases success probability
  - Industry standard (AWS SDK, Google Client Libraries use this)
  - Prevents tight retry loops that worsen rate limiting
- **Trade-offs**:
  - **Benefits**: Higher success rate, respectful to API limits
  - **Compromises**: Longer total wait time in worst case (5s + 10s + 20s = 35s)
- **Follow-up**: Detect rate limit errors from `/impl` command output (parse Claude Code error messages)

## Risks & Mitigations

- **Risk 1: `/impl` command output parsing fragility** — `/impl` command output format may change, breaking error detection logic
  - **Mitigation**: Use conservative pattern matching (grep for "error", "failed", "rate limit"), provide manual retry option if detection fails
- **Risk 2: spec.json schema evolution** — Future kiro updates may change spec.json structure
  - **Mitigation**: Use defensive jq queries with `// default` fallbacks, validate required fields before processing
- **Risk 3: Concurrent executions** — Multiple auto_impl.sh instances may conflict (same spec, file locks)
  - **Mitigation**: Check for existing lockfile (`.kiro/specs/<spec>/.auto_impl.lock`), fail fast if locked, cleanup on EXIT trap
- **Risk 4: Partial task completion** — Script interrupted mid-task leaves spec in inconsistent state
  - **Mitigation**: Log all state transitions, provide `--resume` flag to skip completed tasks (check tasks.md for `[x]` markers)

## References

- [MIT Safe Shell Scripts](https://sipb.mit.edu/doc/safe-shell/) — Defensive bash patterns
- [Bash Select Tutorial](https://linuxize.com/post/bash-select/) — Interactive menu implementation
- [Baeldung Retry Guide](https://www.baeldung.com/linux/shell-retry-failed-command) — Retry mechanism patterns
- [Better Stack Logrotate](https://betterstack.com/community/guides/logging/how-to-manage-log-files-with-logrotate-on-ubuntu-20-04/) — Log management strategies
- [Stack Overflow Config Precedence](https://stackoverflow.com/questions/11077223/what-order-of-reading-configuration-values) — Configuration loading order
