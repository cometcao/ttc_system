########## split files ###########


from os import listdir
from os.path import isfile, join
from pickle import dump
from pickle import load
onlyfiles = [f for f in listdir('./training_data/trained_original_data') if isfile(join('./training_data/trained_original_data', f))]

for file in onlyfiles:
    filename = './training_data/trained_original_data/{0}'.format(file)
    A, B = load(open(filename, 'rb'))
    mid = int(len(A)/2)
    A1 = A[:mid]
    A2 = A[mid:]
    B1 = B[:mid]
    B2 = B[mid:]
    
    path_parts = filename.split("/")
    
    filename1 = '{0}/{1}_1.pkl'.format(path_parts[1], path_parts[-1][:-4])
    filename2 = '{0}/{1}_2.pkl'.format(path_parts[1], path_parts[-1][:-4])
    print(filename1)
    print(filename2)
    dump((A1, B1), open(filename1, 'wb'))
    dump((A2, B2), open(filename2, 'wb'))