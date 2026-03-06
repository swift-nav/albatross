EXTRA_FLAGS = {
    "test_summary": struct(
        command = "common:ci",
        default = "short",
        description = "Sets the level of detail for test summaries.",
    ),
    "lockfile_mode": struct(
        command = "common:ci",
        default = "update",
        description = "We are using update to avoid, that Mac users have to lock the lockfile for linux env",
    ),
    "experimental_fetch_all_coverage_outputs": struct(
        command = "coverage",
        default = False,
        description = "Do not remotely download coverage files for tests.",
    ),
    "show_progress_rate_limit": struct(
        command = "common:ci-sparse",
        default = 60,
        description = "Disable this setting.",
    ),
    "curses": struct(
        command = "common:ci",
        default = "no",
        description = "Disable this setting.",
    ),
}
