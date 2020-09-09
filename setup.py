import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pioneer_das_api", # Replace with your own username
    version="0.0.1",
    author="Leddartech",
    description="Pioneer's das api",
    long_description=long_description,
    long_description_content_type="text/markdown",
    dependency_links=["C:\\Users\\user361\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\leddartech\\common"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True
)