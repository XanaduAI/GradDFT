# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

# Parse requirements from file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

short_description = \
    """
    A library for learning the density functional theory exchange and 
    correlation functional from data.
    """

setup(
    name='grad_dft',
    version='0.1',
    author='Xanadu Quantum Technologies',
    author_email='jack.baker@xanadu.ai',
    description=short_description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/XanaduAi/grad_dft',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "examples": [
            "openfermion>=1.5.1", "tqdm>=4.66.1", "torch>=2.0.1",
            "matplotlib>=3.7.2", "pandas>=2.0.3", "seaborn>=0.12.2",
            "openpyxl"
        ],
    },  
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved ::  Apache 2.0",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
