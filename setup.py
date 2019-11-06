    #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

info = sys.version_info

setup(
    name='pytorch-yanai',
    version='0.0.1',
    description='The sophisticated tool needed for xxxx.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='YouseiTakei',
    author_email='yousei_san@icloud.com',
    url='https://github.com/YouseiTakei/pytorch-yanai',
    packages=['edn', 'pose'],#find_packages(),
    include_package_data=True,
    keywords='pytorch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.5',
        "Operating System :: OS Independent",
    ],
    #entry_points = {
    #    'console_scripts': ['nbc=nbc.nbc:main'],
    #},
    #test_suite="test",
)
