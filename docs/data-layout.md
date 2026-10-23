
## Local dataset setup

The raw datasets are not stored in Git.

Expected paths from the repository root:

- `data/fer/fer2013.csv`
- `data/kdef/KDEF/`

Ensure that only KDEF/ is used for training, and move its folder to expected path.

KDEF contains 4,900 photographs. Its original subject
and acquisition folder structure must be preserved.

The fuzzy classifier accepts an alternative KDEF location
through the `KDEF_ROOT` environment variable.

The original KDEF loader may use
`data/KDEF_and_AKDEF/KDEF/`. A local symlink can provide
that path without duplicating the photographs.
