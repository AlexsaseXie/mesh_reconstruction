import csv
import os

CSV_PATH = 'all.csv'
IDS_ROOT = 'ids/'

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASSES = CLASS_IDS_ALL.split(',')

F = {}

for c in CLASSES:
	f = {}
	f['train'] = open(os.path.join(IDS_ROOT,'%s_train_ids.txt' % c),'w')
	f['test'] = open(os.path.join(IDS_ROOT,'%s_test_ids.txt' % c),'w')
	f['val'] = open(os.path.join(IDS_ROOT,'%s_val_ids.txt' % c),'w')

	F[c] = f


csv_reader = csv.reader(open(CSV_PATH))
for row in csv_reader:
	if row[1] in CLASSES:
		F[row[1]][row[4]].write('%s/%s\n' % (row[1],row[3]))


		
		
		
