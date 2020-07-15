import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textgo",
    version="0.5",
    author="Pairui Li",
    author_email="953383269@qq.com",
    description="Let's go and play with text!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lipairui/textgo",
    #packages=setuptools.find_packages(),
    packages=['textgo','textgo.classifier','textgo.data'],
    install_requires=[ # 依赖列表
        'jieba>=0.42.1',
        'tqdm>=4.46.0',
        'transformers>=2.3.0',
        'numpy>=1.18.4',
        'torch>=1.5.0+cu101',
        'pandas>=1.0.3',
        'gensim>=3.8.3',
        'faiss_cpu>=1.6.3',
        'faiss>=1.5.3',
        'scikit_learn>=0.23.1',
        'xgboost>=1.1.0'
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
