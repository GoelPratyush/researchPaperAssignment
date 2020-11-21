.SILENT:

# Crops images and creates train and test image sets.
train: split_crop.py
	python3 split_crop.py

# Removes all generated directories and files.
clean:
	rm -rf train
	rm -rf test