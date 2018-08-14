from elasticsearch import Elasticsearch, client
import tensorflow as tf
import cv2
import base64
import numpy as np

es = Elasticsearch()

def _get_es_batch(ind = 'open_image', d_type = 'train'):
    name = ind +'_'+d_type
    page = es.search(
            index = name,
            doc_type = d_type,
            scroll = '3m',
            # batch size
            size = 1,
            body = {
                'query':{'match_all':{}}
            })
    sid = page['_scroll_id']
    scroll_size = page['hits']['total']

    while(scroll_size>0):
        page = es.scroll(scroll_id = sid, scroll = '3m')
        sid = page['_scroll_id']
        results = page['hits']['hits']
        scroll_size = len(results)        
        image = base64.b64decode(results[0]['_source']['image'])
        

        yield _transform_image(image)

# make this into image and resize
def _transform_image(im,image_size = [800,800]):
    image_decoded = tf.image.decode_jpeg(im)
    image_resized = tf.cast(tf.image.resize_images(image_decoded,image_size),tf.uint8)

    return image_resized
# call this to get 32 tensorflowed images
def _image_tensors():

    ds = Dataset.from_generator(
        _get_es_batch, (tf.float32), (tf.TensorShape([None])))
    
    yield ds