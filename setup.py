from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

requirements = ['numpy>=1']

setup(
    name='rl',
    version='0.0.0',
    url='https://github.com/baakbrothers/rl',
    author='Yuki Kitayama',
    description='Reinforcement learning package',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements
)
