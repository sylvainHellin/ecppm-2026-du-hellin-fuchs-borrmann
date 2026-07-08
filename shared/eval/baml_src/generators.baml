generator target {
    output_type "python/pydantic"

    // Generated baml_client/ goes to the parent of baml_src/, i.e. shared/eval/
    output_dir "../"

    // Pinned to match baml-py version in shared/eval/pyproject.toml.
    // Update both together (see: pip show baml-py).
    version "0.217.0"

    default_client_mode sync
}
