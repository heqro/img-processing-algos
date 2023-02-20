import _pickr
import multiprocessing

img_indices = list(range(101))
pool = multiprocessing.Pool()
pool.map(_pickr.process_img, img_indices)
