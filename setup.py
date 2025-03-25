from setuptools import setup, find_packages

setup(
    name="ChiefWarden",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'pefile',
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'torch',
        'tqdm',
        'joblib'
    ],
    entry_points={
        'console_scripts': [
            'ChiefWarden=src.__main__:main',
        ],
    },
    include_package_data=True,
)