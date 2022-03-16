from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
	name="censai",
    author="Alexandre Adam",
    author_email="adam.alexandre01123@gmail.com",
    short_description="Free-form gravitational lensing reconstruction of density maps and background intensity maps using a Recurrent Inference Machine."
    long_description=long_description,
    long_description_content_type='text/markdown'
	version="1.0",
	description="",
	packages=find_packages(),
	python_requires=">=3.6"
)
