from setuptools import setup, find_packages

setup(
    name='grad300',
    version='0.1',
    description='Processing pipeline for GRAD-300 data',
    packages=find_packages(),
    install_requires=[
        'astropy',
        'numpy',
        'matplotlib',
        'scipy',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'grad300=grad300.cli:main',
        ],
    },
)
