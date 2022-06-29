import setuptools

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements('requirements.txt')

setuptools.setup(
    name="pioneer_das_api", # Replace with your own username
    version="1.4.2",
    author="Leddartech",
    description="Pioneer's das api",
    packages=[
        'pioneer', 
        'pioneer.das', 
        'pioneer.das.api', 
        'pioneer.das.tests',
        'pioneer.das.api.samples', 
        'pioneer.das.api.samples.annotations',
        'pioneer.das.api.sensors', 
        'pioneer.das.api.sources',
        'pioneer.das.api.datasources',
        'pioneer.das.api.datasources.virtual_datasources',
        'pioneer.das.api.egomotion'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    dependency_links=["https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple/pioneer-common"],
    install_requires=install_reqs,
    include_package_data=True
)