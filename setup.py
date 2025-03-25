from setuptools import setup, find_packages
import os

# Get list of model files to include
model_files = [
    os.path.join('models/default', f) 
    for f in os.listdir('models/default')
    if f.endswith(('.pkl', '.pth'))
]

setup(
    name="ChiefWarden",
    version="1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
            'ChiefWarden=ChiefWarden.__main__:main',
        ],
    },
    include_package_data=True,
    package_data={
        'ChiefWarden': model_files,
    },
    # Exclude these from installation but keep in repo
    exclude_package_data={
        '': ['old_scripts_for_testing_and_training/*', 'LICENSE', 'README.md', 'logo.png'],
    },
)