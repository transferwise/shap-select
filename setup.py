from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="shap-select",
    version="0.1.1",
    description="Heuristic for quick feature selection for tabular regression/classification using shapley values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wise Plc",
    url="https://github.com/transferwise/shap-select",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        "pandas",
        "scipy>=1.8.0",
        "shap",
        "statsmodels",
    ],
    extras_require={
        "test": ["flake8", "pytest", "pytest-cov"],
    },
    packages=find_packages(
        include=["shap_select", "shap_select.*"],
        exclude=["tests*"],
    ),
    include_package_data=True,
    keywords="shap-select",
)
