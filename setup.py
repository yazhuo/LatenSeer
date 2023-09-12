from setuptools import setup, find_packages

setup(
    name='latenseer',
    version='0.1',
    author='Yazhuo Zhang',
    author_email='z.yazhuo@gmail.com',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas',
        'networkx',
        'numpy',
        'tqdm',
        'pyyaml',
        'matplotlib',
        'scipy',
    ],
)