These last two examples are pretty slow and require to have precomputed the molecule in an hdf5 file in advance. Please modify accordingly lines

```python
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dissociation/")
training_files = ["H2_dissociation_omegas.h5"]
```

in those two files. In the case of `train_complex_model.py`, the hdf5 file should have processed the molecules with

```python
from grad_dft.interface.pyscf import molecule_from_pyscf
omegas = jnp.array([0., 0.4])
molecule = molecule_from_pyscf(mf, omegas = omegas)
```

The `test_constraints.py` file executes an example showing that constraints run. However, note that they have not been tested thoroughly.
