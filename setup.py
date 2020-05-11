import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name="GANtools",
        version="0.3.0",
        author="Vee9ahd1",
        author_email="Vee9ahd1@arson.club",
        description="cli tools for image generation w bigGAN",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://gitlab.com/Vee9ahd1/gantools",
        packages=setuptools.find_packages(),
        install_requires=[
            'argparse',
            'requests',
            'pillow',
            'numpy',
            'scipy',
#            'tensorflow-gpu',
#            'tensorflow-hub',
            ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GPL",
            "Operating System :: OS Independent",
            ],# TODO: additional classifiers needed
        entry_points={
            'console_scripts': ['gantools=gantools.cli:main'],
            },
        test_suite='test',
        )
