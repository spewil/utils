from setuptools import setup, find_namespace_packages

# requirements = [
#     "numpy",
#     "opencv-python",
#     "nptdms",
#     "configparser",
#     "pandas",
#     "tqdm",
#     "matplotlib",
#     "seaborn",
#     "scipy",
#     "vtk",
#     "pyyaml",
#     "statsmodels",
#     "requests",
#     "pyexcel",
#     "pyexcel-xlsx",
#     "pyjson",
# ]

setup(
    name='utils',
    version='0.1',
    description="Frequently Used Utilities",
    author='Spencer Wilson',
    author_email='spencer@spewil.com',
    url='www.spewil.com',
    packages=find_namespace_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=False,
    # install_requires=requirements,
)