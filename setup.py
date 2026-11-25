import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opencosmorspy",
    version="0.1.0",
    author="thmsg",
    description="openCOSMO-RS python implementation of COSMO-RS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TUHH-TVT/openCOSMO-RS_py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=["numpy", "scipy", "pandas", "matplotlib"],
    extras_require={"visualization": ["plotly", "rdkit", "scikit-image"]},
)
