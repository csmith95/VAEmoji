import os 

for filename in os.listdir('images'):
	img_num = filename[filename.index('_')+1: -4]
	if " 2.png" in filename:
		print('removing ', filename)
		os.remove('images/{}'.format(filename))