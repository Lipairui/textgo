# coding: utf-8
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textgo",
    version="1.2",
    author="Pairui Li",
    author_email="953383269@qq.com",
    description="Let's go and play with text!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lipairui/textgo",
    packages=setuptools.find_packages(),
    include_package_data = True,
    #packages=['textgo','textgo.classifier'],
    install_requires=[ # 依赖列表
        'jieba>=0.42.1',
        'tqdm>=4.46.0',
        #'transformers>=2.3.0', # not support for python2
        'numpy>=1.16.1',
        'torch>=1.4.0+cu101',
        'pandas>=0.24.0',
        'gensim>=3.8.3',
        #'faiss_cpu>=1.6.3', # installing on python2 causes error, python3 is ok.
        'scikit_learn>=0.20.4',
        'xgboost>=0.80'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    classifiers=(
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
