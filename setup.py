from setuptools import setup, find_packages

setup(
    name='grad_dft',
    version='0.1',
    author='Xanadu Quantum Technologies',
    author_email='author@xanadu.ai',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url='https://github.com/your_username/package_name',
    packages=find_packages(),
    # install_requires=open('requirements.txt').readlines(),
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    #     'Intended Audience :: Developers',
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.6',
    #     'Programming Language :: Python :: 3.7',
    #     'Programming Language :: Python :: 3.8',
    # ],    
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ],
)
