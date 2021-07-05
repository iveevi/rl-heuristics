import os

from datetime import datetime

directory = f'\'results {datetime.now()}\''
def setup(envtests):
    os.system(f'mkdir -p {directory}')

    # Preliminary upload, to make sure we got credentials and everything
    os.system('gdrive upload prod.txt')

    # Create a directory for each envtest
    for envtest in envtests:
        os.system(f'mkdir -p {directory}/{envtest.ename}')

def upload():
    # TODO: use update
    os.system(f'gdrive upload -r {directory}')