"""Install the llm_merging library"""


from setuptools import setup

setup(
    name="llm_merging",
    version=1.0,
    description="starter code for llm_merging",
    install_requires=[
        "torch", "ipdb"
    ],
    packages=["llm_merging"],
)
