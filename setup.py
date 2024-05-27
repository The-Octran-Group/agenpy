from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='agenpy',
   version='0.1.2',
   description='A python package for setting up agentic behavior for LLMs. Includes optimization for large training data, and adherence to applied interactional policies.',
   license="Apache-2.0",
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='Octran Technologies',
   author_email='contact@octran.tech',
   packages=find_packages(), #same as name
   install_requires=["openai>=1.30"], #external packages as dependencies
   python_requires=">=3.8",
)