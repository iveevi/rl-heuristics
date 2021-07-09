import os

from datetime import datetime

directory = f'results {datetime.now()}'.replace(' ', '_')
def setup(envtests):
    os.system(f'mkdir -p {directory}')
    os.system('gdrive about')

    # Create a directory for each envtest
    for envtest in envtests:
        os.system(f'mkdir -p {directory}/{envtest.ename}')

def upload(auto = False, sudo = None):
    if auto:
        os.system(f'gdrive upload -r {directory}')
    elif not sudo == None:
        print('Skipping upload.')

        return
    
    while True:
        str = input('Upload to drive? [y/n] ')

        if str == 'y':
            break
        elif str == 'n':
            return

    os.system(f'gdrive upload -r {directory}')
