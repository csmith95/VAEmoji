from PIL import Image
import requests
import shutil
import os
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

def download_image(url, image_file_path):
	global means, stds

	r = requests.get(url, timeout=4.0)
	if r.status_code != requests.codes.ok:
		return 'Failed'

	stream = io.BytesIO(r.content)
	with Image.open(stream) as image:
		image = image.resize((224, 224))
		# newData = []
		# for item in image.getdata():
		# 	if item[-1] == 0:
		# 		newData.append((255, 255, 255, 255))
		# 	else:
		# 		newData.append(item)

		# image.putdata(newData)
		image.save(image_file_path)

	return 'Success'


try:
	shutil.rmtree('images')
except FileNotFoundError:
	pass # do nothing
os.mkdir('images')

with ThreadPoolExecutor(max_workers = 8) as executor:
	with open('image_urls.json') as f:
		future_to_url = {}
		for i, image_meta in enumerate(json.load(f)):

			url = image_meta['image_url'].strip() # ensure no newline char
			image_file_path = 'images/emoji_{}.png'.format(i)
			future_to_url[executor.submit(download_image, url, image_file_path)] = url

		for future in as_completed(future_to_url):
			url = future_to_url[future]
			try: 
			  data = future.result()
			  print('{}: {}'.format(url, data))
			except Exception as exc:
			  print('{} generated an exception: {}'.format(url, exc))

			


	