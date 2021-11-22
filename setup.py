"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="epi-quark",
    version="0.1.0",
    description="Algorithm agnostic evaluation for (disease) outbreak detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aauss/epi-quark",
    author="Auss Abbood and Stéphane Ghozzi",
    author_email="a.abbood@live.de",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
    ],
    keywords="score, metric, epidemiology, public health, AI",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9, <4",
    install_requires=["scikit-learn>=0.24.2", "pandas>=1.3.0"],
    extras_require={
        "dev": [
            "black>=21.7",
            "codecov",
            "flake8>=4.0.1",
            "isort",
            "mypy>=0.910",
            "nbsphinx>=0.8.7",
            "pre-commit",
            "pydata-sphinx-theme>=0.7.1",
            "pytest>=6.2.4",
            "pytest-cov>=2.12.1",
            "pytest-datadir>=1.3.1",
            "pytest-sugar>=0.9.4",
            "sphinx>=4.2.0",
            "sphinxcontrib-napoleon",
        ]
    },
    #
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    # package_data={  # Optional
    #     "sample": ["package_data.dat"],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[("my_data", ["data/data_file"])],  # Optional
    #
    project_urls={
        "Bug Reports": "https://github.com/aauss/epi-quark/issues",
        "Source": "https://github.com/aauss/epi-quark",
    },
)
