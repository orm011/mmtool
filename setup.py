from setuptools import setup, find_packages

setup(
    name="mmtool",
    version="1.3.0",
    url="git@github.com:orm011/mmtool.git",
    author="Oscar Moll",
    author_email="orm@csail.mit.edu",
    description="some tools to experiment with training multimodal models",
    packages=find_packages(where="mmtool"),
    # install_requires=[],  # install from conda_requirements
    python_requires=">=3.10",
)
