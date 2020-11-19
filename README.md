# iama

[![PyPI version](https://badge.fury.io/py/iama.svg)](https://pypi.org/project/iama/)

_iama_ is a classifier for r/relationships titles. Consider the following title:

```{.bash}
My [18M] girlfriend [18F] is seeing someone else.
```

Here, the post author is an 18-year old male. _iama_ automatically predicts this,
allowing for the generation of a large annotated corpus from r/relationships.

## Installation

```{.bash}
pip install iama
```

## Usage

```python
from iama.model import Iama
iama = Iama()
iama.predict("My [18M] girlfriend [18F] is seeing someone else.")  # "[18M]" returned
```
