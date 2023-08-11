import density_functional_approximation_dm21 as dm21
from openfermion.chem import geometry_from_pubchem
import tensorflow as tf

from pyscf import gto, dft

mol = gto.Mole()
mol.atom = geometry_from_pubchem("water")
mol.basis = "cc-pvdz"
mol.build()

mf = dft.RKS(mol)
mf._numint = dm21.NeuralNumInt(dm21.Functional.DM21)

mf.kernel()
