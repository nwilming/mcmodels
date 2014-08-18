import h5py
import sys
import string
import progressbar
import json
import numpy as np

def fileLineIter(inputFile,
        inputNewline=" ",
        outputNewline=None,
        readSize=8192):
    """Like the normal file iter but you can set what string indicates newline.

    The newline string can be arbitrarily long; it need not be restricted to a
    single character. You can also set the read size and control whether or not
    the newline string is left on the end of the iterated lines.  Setting
    newline to '\0' is particularly good for use with an input file created with
    something like "os.popen('find -print0')".
    """
    if outputNewline is None: outputNewline = inputNewline
    partialLine = ''
    pgress = 0
    while True:
        charsJustRead = inputFile.read(readSize)
        pgress += readSize
        if not charsJustRead: break
        partialLine += charsJustRead
        lines = partialLine.split(inputNewline)
        partialLine = lines.pop()
        for line in lines: yield pgress, line + outputNewline
    if partialLine: yield pgress, partialLine

if __name__ == '__main__':
    filenames = sys.argv[1:]
    with h5py.File('test.hdf5') as hdf5:
        for filename in filenames:
            shapes = json.load(open(filename))
            for name, shape in shapes.iteritems():
                fname = filename.replace('shapes.json', name+'.txt')
                data = hdf5.create_dataset(name, shape=shape)
                print fname, name, shape
                with open(fname) as fob:
                    fob.seek(0,2)
                    size = fob.tell()
                    fob.seek(0)
                    bar = progressbar.ProgressBar(maxval=size).start()
                    for i,(cnt, f) in enumerate(fileLineIter(fob)):
                        index = np.unravel_index(i, shape)        
                        data[index] = string.atof(f)
                        if cnt < size:
                            bar.update(cnt)


