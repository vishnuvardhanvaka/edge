import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import sys
import numpy as np
import math
import time

t1=time.time()

def transform(image):
    trans=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512))
        ])
    return trans(image)

def cvt_pil(image):
    trans=transforms.ToPILImage()
    return trans(image)
def show(image):
    cvt_pil(image).show()

def deep(image,kernel):
    conv=nn.Conv2d(1,1,3,bias=False)
    with torch.no_grad():
        conv.weight=nn.Parameter(kernel)
    return conv(image)
    
def operation(rc,gc,bc,op='su'):
    print(op)
    if op=='su':
        kernel=torch.tensor([[-1,-2,-1],[0,0,0],[-1,-2,-2]],dtype=torch.float)
        kernel=kernel.view(1,1,3,3)
    elif op=='cy':
        print('ok')
        kernel=torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=torch.float)
        kernel=kernel.view(1,1,3,3)

    rcout=deep(rc,kernel)
    bcout=deep(bc,kernel)
    gcout=deep(gc,kernel)    
    final_image=torch.cat([rcout,gcout,bcout],dim=0)
    return rcout

def gaussian_kernel(size,sigma=1):
    print('in blur')
    k=int((size-1)/2)
    kernel=torch.zeros(size,size)
    for i in range(size):
        for j in range(size):
            kernel[i,j]=(1/(2*math.pi*(sigma**2)))*(1/math.exp((((i-(k+1))**2)+((j-(k+1))**2))/(2*(sigma**2))))
    kernel=kernel.view(1,1,5,5)
    return kernel
def cal_gradient(image):
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float).view(1,1,3,3)
    ky=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float).view(1,1,3,3)
    ix=deep(image,kx)
    iy=deep(image,ky)
    ran=ix.shape[1]
    g=torch.zeros(ran,ran)
    theta=torch.zeros(ran,ran)
    print('in grad')
    for i in range(ran):
        for j in range(ran):
            x1=ix[0][i][j]
            x2=iy[0][i][j]
            ang=math.atan(x2/x1)
            r=math.sqrt(x1**2+x2**2)
            g[i,j]=r
            theta[i,j]=ang
    return g,theta

def nm_supression(image,angle):
    m=image.shape[1]
    z=torch.zeros(m,m)
    angle=angle*180./math.pi
    angle[angle<0]+=180
    print('in nm_supression')
    for i in range(m):
        for j in range(m):
            q=255
            r=255
            try:
                #angle 0
                if (0<=angle[i,j]<22.5) or (157.5<=angle[i,j]<180):
                    q=image[i,j+1]
                    r=image[i,j-1]
                #angle 45
                elif (22.5<=angle[i,j]<67.5):
                    q=image[i-1,j+1]
                    r=image[i+1,j-1]
                #angle 90
                elif (67.5<=angle[i,j]<112.5):
                    q=image[i+1,j]
                    r=image[i-1,j]
                #angle 135
                elif (112.5<=angle[i,j]<157.5):
                    q=image[i-1,j-1]
                    r=image[i+1,j+1]
                if (image[i,j]>=q) and (image[i,j]>=r):
                    z[i,j]=image[i,j]
                else:
                    z[i,j]=0
            except IndexError as e:
                continue
    return z
def double_threshold(image,ltr=0.05,htr=0.18):
    print('in double_threshold')
    ht=image.max()*htr
    lt=ht*ltr

    strong=255
    weak=25

    m=image.shape[1]
    res=torch.zeros((m,m),dtype=torch.int32)
    print(ht,lt)

    for i in range(m):
        for j in range(m):
            if image[i,j]>=ht:
                res[i,j]=strong
            elif (image[i,j]>lt) and (image[i,j]<ht):
                res[i,j]=weak
    
    return res,strong,weak
            
def hysterisis(img,strong,weak):
    m=img.shape[1]
    print('in hysterisis')
    for i in range(m):
        for j in range(m):
            if img[i,j]==weak:
                try:
                    if (img[i,j+1]==strong)or(img[i,j-1]==strong)or(img[i+1,j]==strong)or(img[i-1,j]==strong)or(img[i-1,j+1]==strong)or(img[i-1,j-1]==strong)or(img[i+1,j+1]==strong)or(img[i+1,j-1]==strong):
                        img[i,j]=strong
                    else:
                        img[i,j]=0
                except IndexError as e:
                    pass
    print(img,img.shape)
    show(img)
    return img

def canny(image):
    kernel=gaussian_kernel(5,1.4)
    nfi=deep(image,kernel)
    
    grad,theta=cal_gradient(nfi)
    show(grad)
    si=nm_supression(grad,theta)
    show(si)
    ti,strong,weak=double_threshold(si)
    show(ti)
    
    final_image=hysterisis(ti,strong,weak)
    
   
#loading image
pimag=Image.open('kvr.jpg')
pimag=np.array(pimag)

#transfroming image
tra=transform(pimag)
rc=tra[0].repeat(1,1,1)
gc=tra[1].repeat(1,1,1)
bc=tra[2].repeat(1,1,1)
'''
op=input('1.sobel_up == su\n2.sobel_down == sd\n3.sobel_left == sl\n4.sobel_right == sr\n5.canny == cy\n6.gaussian == gu\n\nenter your choice : ')
r=operation(rc,gc,bc,op)
show(r)'''

ca=canny(rc)
t2=time.time()
print(f'time to taken to execute the program is {(t2-t1)/60}mins')























