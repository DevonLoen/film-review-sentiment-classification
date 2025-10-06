from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return a list of requirements from the given file path.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='Sentiment Classification based on film review',
    version='0.1',
    author='Devon Marvellous Loen',
    author_email='devonloen99@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)