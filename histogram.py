import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import math

def transform(image):
    trans=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((512,512))
        ])
    return trans(image)

def cvt_pil(image):
    trans=transforms.ToPILImage()
    return trans(image)
def show(image):
    pil=cvt_pil(image)
    pil.show()

def increase_contrast(image):
    print('increase_contrast')
    img=np.array(image)
    m,n,c=img.shape
    fimg=img.flatten()
    tp=len(fimg)
    print(tp)
    d={}
    pdf={}
    cfd={}
    fd={}
    nop=0
    for i in range(256):
        d[i]=0
    for i in fimg:
        if nop%60000==0:
            print(f'{tp-nop} pixels are remaining')
        d[i]+=1
        nop+=1
    for i in d:
        pdf[i]=d[i]/nop
    for i in pdf:
        if i==0:
            cfd[i]=pdf[i]
        else:
            cfd[i]=pdf[i]+cfd[i-1]
    for i in cfd:
        cfd[i]=cfd[i]*255
        fd[i]=round(cfd[i])
    for i in range(nop):
        fimg[i]=fd[fimg[i]]
    fimg=fimg.reshape(m,n,c)
    fimg=transform(fimg)
    return fimg
def conv(image,kernel):
    print('in_conv')

    conv=nn.Conv2d(1,1,5,bias=False)
    with torch.no_grad():
        conv.weight=nn.Parameter(kernel)
    t=conv(image)
    return t    

def gauss_kernel(size,sigma=1):
    print('in_kernel')
    k=int((size-1)/2)
    kernel=torch.zeros((size,size),dtype=torch.float)
    for i in range(size):
        for j in range(size):
            y=((((i+1)-(k+1))**2)+(((j+1)-(k+1))**2))/(2*(sigma**2))
            z=1/(2*math.pi*(sigma**2))
            kernel[i,j]=z* (1/math.exp(y))
    return kernel

def gauss(image):
    print('gauss')
    kernel=gauss_kernel(5,1.4)
    kernel=kernel.repeat(1,1,1,1)
    si=conv(image,kernel)
    return si

#loading image..........
img=Image.open('images/kvr1.jpg')
#img=img.convert('L')
img.show()


r=increase_contrast(img)
show(r)

sys.exit()
si=gauss(r)
for i in range(0):
    si=gauss(si)
show(si)






















