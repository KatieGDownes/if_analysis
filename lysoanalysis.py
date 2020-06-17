#prototype code for lysosome image analysis (detect, segment and quantify) from IF microscopy data
#created 17th June 2020 by Daniel S. Han

import numpy as np
import cv2
import matplotlib.pyplot as plt

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
def maskOrganelle(im):
    ret,th2 = cv2.threshold(np.array(im,'uint8'),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th3 = cv2.adaptiveThreshold(np.array(im,'uint8'),1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,2)
    th2[th2==0] = 2
    th2 = th2-1
    th2[th3==0] = 0
    return th2

################################################MAIN
folders = ['GFP','GFP-KASH5-DC-DK','GFP-KASH5-DK','GFP-KASH5-DK-fue','GFP-KASH5-DK-mod1']

#create a result file
with open('data_lamp.csv','w') as file:
    file.write('int,r,theta,gfpint,cellnum,exptype\n')
file.close()

for fcount in range(0,len(folders)):
    cid = 0
    for pcount in range(1,11):
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
            print(i)
            #get mask for single contour
            oneoutline = np.zeros_like(gfp_outline)
            cv2.fillPoly(oneoutline, pts =[outcon[i]], color=(1))
            [cc,cr] = getCenter(outcon[i])
            c,r,w,h = cv2.boundingRect(outcon[i])
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
            lysomask = maskOrganelle(seglysopic)
            golmask = maskOrganelle(seggolpic)
            #find the lysosomes that are on and calculate distance from center then save info to file
            (ron,con)=np.nonzero(lysomask)
            ron = ron
            con = con
            rmax = np.amax(np.sqrt(np.power(ron-cr,2)+np.power(con-cc,2)))
#            print(lamp_temp.shape,ron,con)
            #save lyso info
            gfpint = np.sum(gfp_temp*oneoutline)
            with open('data_lamp.csv','a') as file:
                for ondex in range(0,len(ron)):
                    r = np.sqrt(pow(ron[ondex]-cr,2)+pow(con[ondex]-cc,2))/(rmax)
                    theta = np.arctan2((con[ondex]-cc),(ron[ondex]-cr))
                    'int,r,theta,gfpint,cellnum,exptype\n'
                    file.write(str(lamp_temp[ron[ondex]][con[ondex]])+','+str(r)+','+str(theta)+','+str(gfpint)+','+str(cid)+','+folders[fcount]+'\n')
            #move onto next cell id
            cid += 1
            
        #     #plot
        #    plt.figure()

        #    ax1 = plt.subplot2grid((4, 4), (0, 0),colspan=3,rowspan=3)
        #    ax2 = plt.subplot2grid((4, 4), (3, 0))
        #    ax3 = plt.subplot2grid((4, 4), (3, 1))
        #    ax4 = plt.subplot2grid((4, 4), (3, 2))
        #    ax5 = plt.subplot2grid((4, 4), (3, 3))
        #    ax6 = plt.subplot2grid((4, 4), (0, 3))

        #    ax1.get_xaxis().set_visible(False)
        #    ax2.get_xaxis().set_visible(False)
        #    ax3.get_xaxis().set_visible(False)
        #    ax4.get_xaxis().set_visible(False)
        #    ax5.get_xaxis().set_visible(False)
        #    ax6.get_xaxis().set_visible(False)

        #    ax1.get_yaxis().set_visible(False)
        #    ax2.get_yaxis().set_visible(False)
#            ax3.get_yaxis().set_visible(False)
#            ax4.get_yaxis().set_visible(False)
#            ax5.get_yaxis().set_visible(False)
#            ax6.get_yaxis().set_visible(False)
#
#            ax1.imshow(imagetoshow)
#            ax1.plot(ccreal,crreal,'r*')
#            ax1.plot(c,r,'bs')
#            ax1.plot(c+w,r+h,'ro')
#
#            ax2.imshow(lamp_temp*oneoutline)
#
#            ax3.imshow(lysomask)
#            ax3.plot(cc,cr,'r*')
#        #    ax3.colorbar()
#
#            ax4.imshow(gol_temp*oneoutline)
#
#            ax5.imshow(golmask)
#            ax5.plot(cc,cr,'r*')
#
#            ax6.imshow(gfp_temp*oneoutline)
#            ax6.set_title('GFP sum int = '+str(np.sum(seggfppic)))
#            plt.tight_layout()
#            plt.show()
##            plt.savefig(folders[fcount]+'_'+str(pcount).zfill(2)+'_'+str(i)+'.png')