# Project guidance

When asked to conduct automated research, read [docs/research-cycle.md](docs/research-cycle.md)
and drive its CLI. The ontology defines allowed behavior; the agent chooses and
executes permitted steps. Resume existing sessions through `status`.

When developing the package, preserve its standard-library-only runtime and run
`uv run pytest -q`. Research-session restrictions do not prevent authorized changes
to the package implementation or its tests.
