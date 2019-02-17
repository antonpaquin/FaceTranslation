from distutils.core import setup


setup(
    name='FaceTranslation',
    version='0.0.1',
    packages=['FaceTranslation'],
    install_requires=[
        'Pillow',
        'torch',
    ],
)