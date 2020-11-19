import os

import setuptools

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(_THIS_DIR, "README.md"), encoding="utf-8") as f:
    _LONG_DESCRIPTION = f.read().strip()


def main():
    setuptools.setup(
        name="iama",
        version="0.1.2",
        author="Daniel You",
        author_email="daniel.you@jerichoapps.org",
        description="Classifying r/relationships post titles by gender.",
        long_description=_LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://github.com/youdaniel/iama",
        packages=setuptools.find_packages(),
        license="Apache 2.0",
        python_requires=">=3.6",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
        ],
        install_requires=["nltk", "setuptools", "scikit-learn", "numpy"],
    )


if __name__ == "__main__":
    main()
