from pathlib import Path
from setuptools import setup, find_packages


HERE = Path(__file__).parent.absolute()
with (HERE / 'README.md').open('rt') as fh:
    LONG_DESCRIPTION = fh.read().strip()


REQUIREMENTS: dict = {
    'core': [
        'dataclasses;python_version<"3.7"',
        'matplotlib',
        'numpy',
        'pandas',
        'requests',
        'scipy',
        'scikit-learn',
        'torch',
        'torchvision',
    ],
    'donut': [
        'zhusuan @ git+https://github.com/thu-ml/zhusuan.git',
        'tfsnippet @ git+https://github.com/haowen-xu/tfsnippet.git@v0.1.2',
        'donut @ git+https://github.com/korepwx/donut.git',
        'tensorflow<=1.15.1',
    ],
    'lint': [
        'flake8',
    ],
    'test': [
        'coverage[toml]',
        'pytest',
    ],
}
# Test requirements for running deprecated dependencies.
REQUIREMENTS['test36'] = REQUIREMENTS['test'] + REQUIREMENTS['donut']

setup(
    name='ad-toolkit',
    version='1.0.0b0',
    description='Deep Learning Algorithms for anomaly detection',
    author='Pawel Budzynski',
    author_email='pawel.budzynski19@gmail.com',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/pbudzyns/ad_toolkit',
    packages=find_packages(),
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=REQUIREMENTS['core'],
    extras_require={
        **REQUIREMENTS,
        'all': [req for reqs in REQUIREMENTS.values() for req in reqs],
    },
)
