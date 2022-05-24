import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opencosmorspy",
    version="0.0.1",
    author="thmsg",
    description="openCOSMO-RS python implementation of COSMO-RS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TUHH-V8/openCOSMO-RS_py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL v2.0",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
