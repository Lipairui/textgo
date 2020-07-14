import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textgo",
    version="0.1",
    author="Pairui Li",
    author_email="953383269@qq.com",
    description="Let's go and play with text!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lipairui/textgo",
    packages=setuptools.find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
        ],
    },
    classifiers=(
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
