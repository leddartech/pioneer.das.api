import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name="pioneer_das_api", # Replace with your own username
    version="0.1.0",
    author="Leddartech",
    description="Pioneer's das api",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['pioneer', 'pioneer.das', 'pioneer.das.api'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'opencv-python'
    ],
    include_package_data=True
)