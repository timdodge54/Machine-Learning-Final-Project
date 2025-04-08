from setuptools import setup, find_packages

setup(
    name='machine_learning_final_project',  # package name (avoid hyphens)
    version='0.1.0',
    description='A machine learning project with various forecasting models.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),  # automatically discover all packages and sub-packages
    install_requires=[
        # List your project's dependencies here.
        # e.g., 'numpy>=1.18.0', 'tensorflow>=2.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
