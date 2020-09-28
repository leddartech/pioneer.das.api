# pioneer.das.api

pioneer.das.api is a python library that provides an api to read and to transform the raw data of leddartech' datasets.

## Installation

Before installing, you should add to your pip.conf file the gitlab pypi server url to trust.

```conf
[global]
extra-index-url = https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple
                  https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/487/packages/pypi/simple
trusted-host = svleddar-gitlab.leddartech.local
```

Use the package manager [pip](https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/487/packages/pypi/simple) to install pioneer.das.api .

```bash
pip install pioneer-das-api
```

When developing, you can link the repository to your python site-packages and enable hot-reloading of the package.
```bash
python3 setup.py develop --user
```

If you don't want to install all the dependencies on your computer, you can run it in a virtual environment
```bash
pipenv install --skip-lock

pipenv shell
```

## Usage

```python
from pioneer.das.api.platform import Platform

pf = Platform('path/to/dataset')

pixell = pf.sensors['pixell_bfc']

echoes = pixell['ech']

```
