import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='myriad',
    version='0.0.1',
    author='Max Halford',
    author_email='maxhalford25@gmail.com',
    description='Multiclass classification with many classes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MaxHalford/myriad',
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn',
        'svmloader'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',
)
