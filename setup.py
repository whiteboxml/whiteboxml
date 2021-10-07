import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

requirements = [
    'matplotlib',
    'scikit-learn',
    'seaborn',
    'setuptools',
    'wheel',
]

setuptools.setup(
    name='whiteboxml',
    version='0.0.2',
    author='WhiteBox',
    author_email='info@whiteboxml.com',
    description='Fancy data functions that will make your life as a data scientist easier.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/whiteboxml/whiteboxml',
    project_urls={
        'Website': 'https://whiteboxml.com',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6',
    install_requires=requirements,
)
