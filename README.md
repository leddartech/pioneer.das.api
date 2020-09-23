# pioneer.common

pioneer.das.api is a python library with 

## Installation

Before installing, you should add to your pip.conf file the gitlab pypi server url to trust.

```conf
[global]
extra-index-url = https://__token__:qcnZ-LPju8cqtpG1cpss@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple
    https://__token__:qcnZ-LPju8cqtpG1cpss@svleddar-gitlab.leddartech.local/api/v4/projects/487/packages/pypi/simple
trusted-host = svleddar-gitlab.leddartech.local
```

Use the package manager [pip](https://__token__:<your_personal_token>@svleddar-gitlab.leddartech.local/api/v4/projects/487/packages/pypi/simple) to install pioneer.common .

```bash
pip install pioneer-common --index-url https://__token__:<your_personal_token>@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple --trusted-host svleddar-gitlab.leddartech.local
```

For development only, you can link the repository to your python site-packages
```bash
python3 setup.py develop --user
```

If you don't want to install all the dependencies on your computer, you can run it in a virtual environment
```bash
pipenv install

pipenv shell
```

## Usage

```python
from pioneer.das.api.platform import Platform

pf = Platform('path/to/dataset')

pixell = pf.sensors['pixell_bfc']

echoes = pixell['ech']

```
