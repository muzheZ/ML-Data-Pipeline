from setuptools import setup

# This is the setup script for the ML pipeline library.
# It defines the package metadata and dependencies.
# 
# Build the package:
#   python3 setup.py bdist_wheel
#
# Install the package:
#   pip install dist/ml_pipeline-1.0.0-py3-none-any.whl

setup(
    name="ml-pipeline",
    version="1.0.0",
    author="<Author Name>",
    author_email="<author email address>",
    description="ML pipeline library",
    packages=[
        "ml_pipeline",
        "ml_pipeline.datasets",
        "ml_pipeline.models",
        "ml_pipeline.mixins",
    ],
    install_requires=[
        "joblib==1.2.0",
        "matplotlib==3.6.3",
        "numpy==1.24.1",
        "omegaconf==2.3.0",
        "pandas==1.5.3",
        "scikit-learn==1.2.1",
        "seaborn==0.12.2",
    ],
)