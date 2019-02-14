"""Setup file."""

from distutils.core import setup

setup(
    name='dcspn',
    version='0.1',
    description='DCSPN Library',
    author='Jhonatan S. Oliveira, Andre E. dos Santos, Andre L. Teixeira,\
            Cory J. Butz',
    author_email='jhonatanoliveira@gmail.com, \
                  andre.eds@gmail.com,\
                  andreloboteixeira@gmail.com\
                  cory.butz@gmail.com',
    packages=['dcspn'],
    py_modules=['tensorflow', 'numpy']
)
