.SILENT:

# Calls make in data.
train:
	cd data && make train

# Cleans all generated directories and files.
clean:
	cd data && make clean