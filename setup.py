from setuptools import setup, find_packages

setup(
    name="drone_audition",
    version="0.0.1",
    author="Dmitrii Mukhutdinov",
    author_email="d.mukhutdinov@qmul.ac.uk",
    description="Drone ego-noise modeling and removal using JAX",
    packages=find_packages(),
)