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

	# replace transparent pixels with white pixels
	# https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
	with Image.open(io.BytesIO(r.content)) as image:
		image = image.resize((256, 256)).convert("RGBA")
		new_image = Image.new("RGBA", image.size, "WHITE")
		new_image.paste(image, (0, 0), image)
		new_image.save(image_file_path)

	return 'Success'


try:
	shutil.rmtree('images')
except FileNotFoundError:
	pass # do nothing
os.mkdir('images')

with ThreadPoolExecutor(max_workers = 6) as executor:
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

			


	