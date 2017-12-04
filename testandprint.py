import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import mxnet as mx 
from Symbol.symbol import get_resnet_model
from Symbol.symbol import YOLO_loss
from data_ulti import get_iterator
import cv2 
# import random

def decodeBox(yolobox, size, dscale):
    i, j, cx, cy, w, h, cls1, cls2, cls3, cls4 = yolobox
    cxt = j*dscale + cx*dscale
    cyt = i*dscale + cy*dscale
    wt = w*size
    ht = h*size
    cls = np.argmax([cls1, cls2, cls3, cls4])
    return [cxt, cyt, wt, ht, cls]

def drawResult(img, label, img_name, dscale=32):
    #assert label.shape == (7,7,9)
    size = img.shape[1]
    ilist, jlist = np.where(label[:,:,0]>0.5)
    
    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(np.uint8(img))
    for i,j in zip(ilist, jlist): 
        cx,cy,w,h,cls1, cls2, cls3, cls4 = label[i,j,1:]
        cxt, cyt, wt ,ht, cls = decodeBox([i, j, cx,cy,w,h,cls1, cls2, cls3, cls4], size, dscale)
        # Create a Rectangle patch
        rect = patches.Rectangle((cxt-wt/2,cyt-ht/2), wt,ht,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    
        name="unkown"
        if cls==0:
            name="car"
        elif cls==1:
            name="pedestrian"
        elif cls==2:
            name="cyclist"
        elif cls==3:
            name="traffic lights"
        plt.text(x=int(cxt-wt/2), y=int(cyt-ht/2), s=str(name), bbox=dict(facecolor='red', alpha=0.5))
    plt.savefig('./result/' + img_name + '.jpg', dpi = 600)

def main():
	# 70091 -- 72090
	# choose randomly 30 images 
	img_list = np.random.randint(70091, 72091, 30).astype('str')
	H, W = 224, 224

	sym, args_params, aux_params = mx.model.load_checkpoint('drive_full_detect', 800)
	logit = sym.get_internals()['logit_output']
	mod = mx.mod.Module(symbol=logit, context=mx.cpu(0))
	
	for img in img_list:
		img_n = './DATA/testing/testing/' + img + '.jpg'
		cat = plt.imread(img_n)
		cat_resize = cv2.resize(cat, (W,H))
		cat_nd = mx.nd.array(ctx=mx.cpu(0), source_array=cat_resize.transpose((2,0,1)).reshape(1,3,H,W))
		cat_itr = mx.io.NDArrayIter(data=cat_nd, data_name='data',  batch_size=1)		
		mod.bind(cat_itr.provide_data)
		mod.init_params(allow_missing=False, arg_params=args_params, aux_params=aux_params, 
                initializer=mx.init.Xavier(magnitude=2,rnd_type='gaussian',factor_type='in'))
		out = mod.predict(eval_data=cat_itr, num_batch=10)
		pred = (out.asnumpy()[0]+1)/2
		drawResult(cat_resize, pred, img)



if __name__ == '__main__':
	main()
