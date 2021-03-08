# pioneer.das.api

pioneer.das.api is a python library that provides an api to read and to transform the raw data of leddartech' datasets. 

## Pixset Dataset
[Pixset](https://leddartech.com/dataset) is the first dataset using the leddartech Pixell sensor. A solid state flash LiDAR that can provide full wave-form data. All the annotated frames of the dataset have been recorded in Montreal and Quebec city under various environmental conditions. 

A full description of the Pixset dataset can be found here: []()

We've also published a set of tools to help users in manipulating the dataset data. The das.api can be used to process data of one or many parts of the dataset with the help of a convenient and user-friendly python api.

The full documentation for the das.api can be found here: [https://leddartech.github.io/pioneer.das.api/](https://leddartech.github.io/pioneer.das.api/)

## Installation

You can install the pioneer.das.api with the package manager pip.


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

You can find more in-depth examples of different use of cases for the pioneer.das.api here: https://github.com/leddartech/pioneer.das.api/tree/master/docs/jupyterNotebooks

