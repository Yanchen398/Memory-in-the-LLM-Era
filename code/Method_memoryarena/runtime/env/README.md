# Environments

Environment implementations are under `env_systems/`. Add task data and
task-specific utilities to the corresponding directory:

- `web_search_env/`
- `travel_planner_env/`
- `web_shopping_env/`
- `formal_reasoning_env/`

Document required services and data preparation in
`setup_<environment>.md`. Keep machine-specific paths in local configuration
files rather than source code.
