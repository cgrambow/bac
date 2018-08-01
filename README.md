# bac
Fit bond additivity corrections.

- Type 1: Linear regression with bond (or atom) types as parameters
- Type 2: Non-linear regression based on BACs described by Anantharaman and Melius, JPCA (2005)

# Input files
Required: Two space-separated `.csv` files with SMILES or InChI in the first column and enthalpies of formation in the second column. The first file contains calculated data and the second file contains experimental data.

Optional: A `.xyz` file with molecular geometries in the same order as the identifiers in the calculated data file.

# How to run
`python fit_bac.py <calc_data> <expt_data> <out_dir>`

# Optional arguments
General:
- `--weighted`: Run a weighted regression if experimental uncertainties are available as a third column in `expt_data`
- `--val_split <float>`: Separate out the specified fraction of data as a validation set
- `--folds <int>`: Run cross-validation (overrides `val_split`)

Specific to type 1:
- `--atom_features`: Fit atom additivity corrections instead of bond additivity corrections

Specific to type 2:
- `--geos <xyz>`: Fit type 2 BACs using the provided xyz file with geometries
- `--geo_exceptions <file>`: Override the geometry check for the identifiers in this file (e.g., when bonds aren't recognized in the 3D geometry)
- `--with_mult`: Also fit molecular corrections if given the multiplicities as a third column in `calc_data` (destroys size consistency)
- `--global_min`: Use a basin hopping algorithm to find the global minimum
- `--global_min_iter <int>`: Number of basin hopping iterations
- `--lam <float>`: Regularization parameter
