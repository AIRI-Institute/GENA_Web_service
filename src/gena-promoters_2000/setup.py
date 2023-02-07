from setuptools import setup

setup(
    name='promoters_service',
    version='0.0.1',
    packages=['src', 'src.gena_lm', 'src.gena_lm.tokenizers', 'src.gena_lm.genome_tools'],
    url='https://gitlab.2a2i.org/bioinformatic/dnalm_service',
    license='HZ',
    author='mks',
    author_email='petrov@airi.net',
    description='Service for bioinformatics annotations fasta files. The 2000 model is used.'
)
