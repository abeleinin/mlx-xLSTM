from setuptools import setup, find_packages

setup(
    name="mlx_xlstm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        "mlx>=0.12.0",
    ],
    author="Abe Leininger",
    author_email="awl21802@gmail.com",
    description="MLX implementation of xLSTM model by Beck et al. (2024)",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/abeleinin/mlx-xLSTM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
