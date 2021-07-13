import os

from datetime import datetime

def setup(enames):
    directory = f'results {datetime.now()}'.replace(' ', '_')

    os.system(f'mkdir -p {directory}')
    os.system('gdrive about')

    # Create a directory for each envtest
    for ename in enames:
        os.system(f'mkdir -p {directory}/{ename}')
    
    return directory

def upload(dir, auto = False, sudo = None):
    if auto:
        os.system(f'gdrive upload -r {dir}')
    elif not sudo == None:
        print('Skipping upload.')

        return
    
    while True:
        str = input('Upload to drive? [y/n] ')

        if str == 'y':
            break
        elif str == 'n':
            return

    os.system(f'gdrive upload -r {dir}')
