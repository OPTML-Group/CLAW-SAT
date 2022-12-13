from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
        "fire",
        "graphviz",
        "jsbeautifier==1.14.0",
        "jsonlines==3.0.0",
        "pyjsparser==2.7.1",
        "tqdm",
        "requests",
        "regex==2022.1.18",
        "loguru",
        "pyarrow",

        # Data
        "matplotlib",
        "numpy",
        "pandas==1.4.0",
        "seaborn==0.11.2",
        # "spacy==2.1.0",

        # PyTorch
        "pytorch-lightning==1.5.10",
        "torch==1.7.0",
        "torchtext==0.8.0",
        "horovod==0.23.0",

        # NLP dependencies
        "sentencepiece==0.1.96",
        "sacremoses==0.0.47",
        "transformers==4.1",
        "tokenizers==0.9.4",
        "datasets",
        "allennlp==2.4.0",
    ],
    extras_require={"test": ["pytest"]}
)
