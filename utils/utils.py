import argparse
import json
from typing import Optional
from utils import DType,  default_dtype
import jax.numpy as jnp

'''class Utils():

    def __init__(self, config_path: str = ''):

        if config_path != '':
            try:
                f = open(config_path)
                f.close()
            except IOError:
                print('<!> Info: No configuration file')
                raise Exception('It is necessary to create a configuration file (.json) for some variables')

            with open(config_path) as json_file:
                    self.config_variables = json.load(json_file)

    def get_config_variables(self):
        return self.config_variables

    def parse_arguments(self):
        """This function parses input to the main program.
        """
        parser = argparse.ArgumentParser(description="Replication of the DM21 paper\n Example: python main.py prediction")

        def boolean_string(s):
            if s not in {'False', 'True'}:
                raise ValueError('Not a valid boolean string')
            return s == 'True'

        parser.add_argument("mode", default="predict", help="Select one mode from training/evaluation/prediction/test", type=str)

        self.args = parser.parse_args()
        return self.args
'''    
def to_device_arrays(*arrays, dtype: Optional[DType] = None):

    if dtype is None:
        dtype = default_dtype()

    return [jnp.asarray(array, dtype=dtype) for array in arrays]