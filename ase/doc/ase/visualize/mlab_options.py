# creates: mlab_options.txt
import subprocess
subprocess.check_call('python3 -m ase.visualize.mlab -h > mlab_options.txt',
                      shell=True)
