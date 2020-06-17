#prototype code for golgi image analysis (detect, segment and quantify) from IF microscopy data
#created 17th June 2020 by Daniel S. Han

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import path
import scipy.interpolate as scinterp
from scipy import fftpack
from skimage import measure

################################################function for finding contours and give back the correct ones
def getContours(gfp_outline):
    #get contours
    ret,thresh = cv2.threshold(gfp_outline,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    outcon = []
    #find all contours from outline image and filter for the proper ones
    for i in range(0,len(contours)):
        area = cv2.contourArea(contours[i])
        if hierarchy[0,i,3] == -1 and area > 1000:
            if area/cv2.arcLength(contours[i],True)>50:
                outcon.append(contours[i])
    return outcon

################################################function for getting center of contour
def getCenter(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY]

################################################function for masking lysosomes
def maskOrganelle(im,nsize):
    ret,th2 = cv2.threshold(np.array(im,'uint8'),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th3 = cv2.adaptiveThreshold(np.array(im,'uint8'),1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,2)
    th2[th2==0] = 2
    th2 = th2-1
    th2[th3==0] = 0
    return th2

################################################function for detecting particles
#function to return contours from the image
def detectParticles(img,nperc):
    perc = 0.1
    imr,imc = img.shape
    #interpolate 3d image
    zim = img.copy()
    r,c = zim.shape
    x = np.zeros_like(zim)
    y = np.zeros_like(zim)
    for i in range(0,r):
        y[i,:] = i
    for i in range(0,c):
        x[:,i] = i
    nskip = int(imr*imc*nperc)
    xsparse = x.flatten()[::nskip]
    ysparse = y.flatten()[::nskip]
    print('length of sparse:',len(xsparse))
    xnew = x
    ynew = y
    zsparse = zim.flatten()[::nskip]
    ######Radial basis function interpolation
    rbf = scinterp.Rbf(xsparse,ysparse,zsparse,epsilon=(perc*imr+perc*imc)/2,smooth=(perc*imr+perc*imc)/2)
    zinterp = rbf(xnew,ynew)
    #process image for blob detection
    img2 = zim-zinterp
    img2[img2<0] = 0
    # img2 += np.abs(np.amin(img2))
    # img2 *= 255./np.amax(img2)
    #image fft filtering
    imfft = fftpack.fft2(img2)
    #keep only fraction of coefficients
    keep_frac = 0.25
    imfft2 = imfft.copy()
    [row,col] = imfft2.shape
    #high frequency cut out
    imfft2[int(row*keep_frac):int(row*(1-keep_frac)),:] = 0
    imfft2[:,int(col*keep_frac):int(col*(1-keep_frac))] = 0
    #inverse fft to remake image
    im_ifft = fftpack.ifft2(imfft2).real
    im_ifft[im_ifft<0] = 0
    #contour detection on fft of image
    imfftth = im_ifft.copy()
    contoursfft = measure.find_contours(imfftth,np.mean(imfftth)+np.std(imfftth))
    img2 = np.array(img2)
    return img2,rbf,contoursfft

################################################MAIN
folders = ['GFP','GFP-KASH5-DC-DK','GFP-KASH5-DK','GFP-KASH5-DK-fue','GFP-KASH5-DK-mod1']

#create a result file
with open('data_golgi.csv','w') as file:
    file.write('int,r,theta,cellnum,exptype\n')
file.close()

for fcount in range(0,len(folders)):
# for fcount in range(0,1):
    cellcount = 0
    for pcount in range(1,11):
    # for pcount in range(1,2):
        #load images
        print(folders[fcount],str(pcount).zfill(2))
        combine_original = cv2.imread('./'+folders[fcount]+'/Color Combine-'+str(pcount).zfill(2)+'.tif',-1)
        gfp = cv2.imread('./'+folders[fcount]+'/GFP-'+str(pcount).zfill(2)+'.tif',-1)
        gfp_outline = cv2.imread('./'+folders[fcount]+'/GFP-'+str(pcount).zfill(2)+'_outlines.tif',0)
        gol = cv2.imread('./'+folders[fcount]+'/GM130-'+str(pcount).zfill(2)+'.tif',-1)
        lamp = cv2.imread('./'+folders[fcount]+'/LAMP-'+str(pcount).zfill(2)+'.tif',-1)
        #function to calculate contours of gfp outline images
        outcon = getContours(gfp_outline)
        #cycle through each contour
        for i in range(0,len(outcon)):
        # for i in range(0,1):
            print(i)
            #get mask for single contour
            oneoutline = np.zeros_like(gfp_outline)
            cv2.fillPoly(oneoutline, pts =[outcon[i]], color=(1))
            [cc,cr] = getCenter(outcon[i])
            # (x,y),(MA,ma),angle = cv2.fitEllipse(outcon[i])
            (x,y),maxradius = cv2.minEnclosingCircle(outcon[i])
            c,r,w,h = cv2.boundingRect(outcon[i])
            ccreal = cc
            crreal = cr
            cc = cc - c
            cr = cr - r
            #make temp pics only for region of contours
            combine = np.copy(combine_original)
            imagetoshow = cv2.drawContours(combine, [outcon[i]], 0, (0,255,0), 3)
            lamp_temp = lamp[r:r+h,c:c+w]
            gol_temp = gol[r:r+h,c:c+w]
            gfp_temp = gfp[r:r+h,c:c+w]
            oneoutline = oneoutline[r:r+h,c:c+w]
            seglysopic = oneoutline*lamp_temp*(255.0/np.amax(oneoutline*lamp_temp))
            seggolpic = oneoutline*gol_temp*(255.0/np.amax(oneoutline*gol_temp))
            seggfppic = oneoutline*gfp_temp
            #analysis organelles
            lysomask = maskOrganelle(seglysopic,17)
            # golmask = maskOrganelle(seggolpic,nsizes[fcount])
            img2,rbf,golcontours = detectParticles(seggolpic,0.001)
            #check golgi mask
            # plt.figure()
            # plt.imshow(gol_temp)
            # for c in golcontours:
            #     plt.plot(c[:,1],c[:,0],'r',linewidth=1)
            # plt.show()
            # plt.savefig("plot_"+folders[fcount]+"_"+str(pcount)+"_"+str(i)+".png",dpi=300)
            # plt.close()
            #take golgi contours and find center of mass
            mask_golgi = np.zeros_like(seggolpic)
            for cnt in golcontours: 
                c2in = []
                for icn in cnt:
                    c2in.append([int(icn[1]),int(icn[0])])
                c2in = [np.array(c2in,dtype='int')]
                cv2.fillPoly(mask_golgi,c2in,color=(1))
        
            seggolpic[mask_golgi==0] = 0
            nz_index = np.transpose(np.nonzero(seggolpic))
            com = [0,0]
            for ind in nz_index:
                # mask_golgi[i[0],i[1]] += 1
                com[0] += seggolpic[ind[0],ind[1]]*ind[0]
                com[1] += seggolpic[ind[0],ind[1]]*ind[1]
            com[0] = com[0]/np.sum(seggolpic)
            com[1] = com[1]/np.sum(seggolpic)
            #find the profile of golgi distribution from center of mass
            with open("data_golgi.csv",'a') as f:
                for ind in nz_index:
                    r = np.sqrt(pow(ind[0]-com[0],2)+pow(ind[1]-com[1],2))/(2*maxradius)
                    theta = np.arctan2((ind[1]-com[1]),(ind[0]-com[0]))
                    f.write(str(seggolpic[ind[0],ind[1]])+","+str(r)+","+str(theta)+","+str(cellcount)+","+folders[fcount]+"\n")
            cellcount+=1
            
            #plot results
            # plt.figure()
            # plt.subplot(131)
            # plt.imshow(gol_temp)
            # plt.subplot(132)
            # plt.imshow(seggolpic)
            # plt.plot(com[1],com[0],'r*')
            # plt.subplot(133)
            # plt.imshow(mask_golgi)
            # plt.plot(com[1],com[0],'r*')
            # # plt.show()
            # plt.savefig("plot_"+folders[fcount]+"_"+str(pcount)+"_"+str(i)+".png",dpi=300)
            # plt.close()