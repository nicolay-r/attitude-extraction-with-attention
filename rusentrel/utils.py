import glob
import os
import shutil


def rm_dir_contents(dir_path):
    contents = glob.glob(dir_path)
    for f in contents:
        print "Removing old file/dir: {}".format(f)
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f, ignore_errors=True)