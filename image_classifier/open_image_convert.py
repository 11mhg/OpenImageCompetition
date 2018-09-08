from classifier_data.classifier_data import *



ppd = PreProcessData('./OpenImage.txt')
ppd.get_open_images('../Dataset/OpenImage/',data_type='test')
ppd.write_tf('../Dataset/OpenImage/classification/',num_shards=5)

ppd.get_open_images('../Dataset/OpenImage/',data_type='val')
ppd.write_tf('../Dataset/OpenImage/classification/',num_shards=5)

ppd.get_open_images('../Dataset/OpenImage/',data_type='train')
ppd.write_tf('../Dataset/OpenImage/classification/',num_shards=15)

print("Done")
