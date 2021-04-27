import setuptools

setuptools.setup(
    name="sghmczpz", # Replace with your own username
    version="0.0.2",
    author="Fan Zhu, Lingyu Zhou, Tiancheng Pan",
    author_email="fan.zhu@duke.edu, lingyu.zhou@duke.edu, tiancheng.pan@duke.edu",
    description="Implementation of Stochastic Gradient Hamiltonian Monte Carlo.",
    url="https://github.com/TianchengPAN/STA-663-Final-Project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "autograd >= 1.2",
        "numpy >= 1.14.2",
        "seaborn >= 0.8.1",
        "matplotlib >= 2.0.0",
        "scipy >= 1.0.1",
        "numba >= 0.37.0"
    ],
)
