import os

from datetime import datetime

def setup(enames):
    directory = f'results {datetime.now()}'.replace(' ', '_')

    os.system(f'mkdir -p {directory}')
    ret = os.system('gdrive about')

    if ret != 0:
        print('Error with gdrive (probably not installed)')
        exit(-1)

    # Create a directory for each envtest
    for ename in enames:
        os.system(f'mkdir -p {directory}/{ename}')

    return directory

def upload(dirn, auto = False, sudo = None):
    if auto:
        os.system(f'gdrive upload -r {dirn}')
    elif not sudo == None:
        print('Skipping upload.')

        return

    while True:
        str = input('Upload to drive? [y/n] ')

        if str == 'y':
            break
        elif str == 'n':
            return

    ret = os.system(f'gdrive upload -r {dirn}')
    if ret != 0:
        print('Error with gdrive (probably not installed)')
        exit(-1)
