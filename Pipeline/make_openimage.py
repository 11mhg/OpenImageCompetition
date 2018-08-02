from data.data import *

d = PreProcessData('./OpenImage.txt')
d.get_open_images('../Dataset/OpenImage/',data_type='val')
d.write_tf('../Dataset/OpenImage/tfrecords/',num_shards=10)

print("Done writing validation")

d.get_open_images('../Dataset/OpenImage/',data_type='test')
d.write_tf('../Dataset/OpenImage/tfrecords/',num_shards=10)

print("Done writing testing")

d.get_open_images('../Dataset/OpenImage/',data_type='train')
d.write_tf('../Dataset/OpenImage/tfrecords/',num_shards=15)

print("Done writing training")


