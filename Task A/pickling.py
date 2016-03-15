import cPickle

#Routine for saving objects in pickle format
def pickleSave(filename, object):
    f = open(filename,"w")
    cPickle.dump(object, f)
    f.close()

#Routine for loading objects in pickle format
def pickleLoad(filename):
    f = open(filename, "r")
    file = cPickle.load(f)
    f.close()
    return file
