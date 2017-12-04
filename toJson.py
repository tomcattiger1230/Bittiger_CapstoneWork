import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx 
from Symbol.symbol import get_resnet_model
from Symbol.symbol import YOLO_loss
from data_ulti import get_iterator
import cv2 
import json

def decodeBox(yolobox, size, dscale):
    i, j, cx, cy, w, h, cls1, cls2, cls3, cls4 = yolobox
    cxt = j*dscale + cx*dscale
    cyt = i*dscale + cy*dscale
    wt = w*size
    ht = h*size
    l_t = [cxt-wt/2., cyt-ht/2.]
    r_b = [cxt+wt/2., cyt+ht/2.]
    cls = np.argmax([cls1, cls2, cls3, cls4])
    if cls == 0:
        cls_i = 1
    elif cls == 1:
        cls_i = 2
    elif cls == 2:
        cls_i = 3
    elif cls == 3:
        cls_i = 20

    return l_t + r_b + [cls_i, str([cls1, cls2, cls3, cls4][cls])]


def json_list(img, label, key_n, dscale=32):
	size = img.shape[1]
	ilist, jlist = np.where(label[:,:,0]>0.5)
	temp_dict = {}
	for i,j in zip(ilist, jlist):
		cx,cy,w,h,cls1, cls2, cls3, cls4 = label[i,j,1:]
		temp_dict[key_n] = decodeBox([i, j, cx,cy,w,h,cls1, cls2, cls3, cls4], size, dscale)
	return temp_dict


def main():
	# 70091 -- 72090 
	img_list = np.arange(70091, 72091).astype('str')
	H, W = 224, 224
	with open ('./result.json', 'w') as f:

		sym, args_params, aux_params = mx.model.load_checkpoint('drive_full_detect', 800)
		logit = sym.get_internals()['logit_output']
		mod = mx.mod.Module(symbol=logit, context=mx.cpu(0))
		# result_dict = {}
		for img in img_list:
			img_n = './DATA/testing/testing/' + img + '.jpg'
			key_n = img+'.jpg'
			cat = plt.imread(img_n)
			cat_resize = cv2.resize(cat, (W,H))
			cat_nd = mx.nd.array(ctx=mx.cpu(0), source_array=cat_resize.transpose((2,0,1)).reshape(1,3,H,W))
			cat_itr = mx.io.NDArrayIter(data=cat_nd, data_name='data',  batch_size=1)		
			mod.bind(cat_itr.provide_data)
			mod.init_params(allow_missing=False, arg_params=args_params, aux_params=aux_params, 
	                initializer=mx.init.Xavier(magnitude=2,rnd_type='gaussian',factor_type='in'))
			out = mod.predict(eval_data=cat_itr, num_batch=10)
			pred = (out.asnumpy()[0]+1)/2
			json.dump(json_list(cat_resize, pred, key_n), f)





if __name__ == '__main__':
	main()
