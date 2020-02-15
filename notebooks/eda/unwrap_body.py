# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import img_as_float, color, exposure
from skimage.restoration import unwrap_phase
import cv2

DIRcorpos = 'C:/Users/G1546 FIRE V3/Desktop/SHEAROGRAFIA/Teste 21-10-2019'
DESTcorpos = 'C:/Users/G1546 FIRE V3/Desktop/SHEAROGRAFIA/Teste 21-10-2019-PhaseXYunwrapped'

if os.path.exists(DESTcorpos) == 0:
    os.mkdir(DESTcorpos)
listaIMAGENS = os.listdir(DIRcorpos)
numIMAGENS = len(listaIMAGENS)



for i in range(0,numIMAGENS):
    frame = cv2.imread(''.join([DIRcorpos,'//',listaIMAGENS[i]]),0)
    #    img = frame[624:1524,616:1516] #Removing parts without interference, register with ultrasound
    #    img = frame #Removing parts without interference, register with ultrasound
    image = color.rgb2gray(img_as_float(frame[300:520,190:450]))
    imageRESCALE = exposure.rescale_intensity(image, out_range=(0, 2 * np.pi))
    #imageRESCALEmax = np.amax(imageRESCALE)
    #imageRESCALEmin = np.amin(imageRESCALE)
    
    # Filter SINE and COSINE
    SENO = np.sin(imageRESCALE)
    COSSENO = np.cos(imageRESCALE)     
    
        #senoMIN = np.amin(SENO)
        #cossenoMAX = np.amax(COSSENO)
        #cossenoMIN = np.amin(COSSENO)
    filterSIZE = 3
    iterations = 5
    sigma = 2    

    for iter in range(0,iterations):
    #        SENO = cv2.medianBlur(SENO,filterSIZE)
        SENO = cv2.GaussianBlur(SENO,(filterSIZE,filterSIZE),sigma)
        #SENOfilteredMAX = np.amax(SENOfiltered)
        #SENOfilteredMIN = np.amin(SENOfiltered)
    #        COSSENO = cv2.medianBlur(COSSENO,filterSIZE)
        COSSENO =  cv2.GaussianBlur(COSSENO,(filterSIZE,filterSIZE),sigma)
        #COSSENOfilteredMAX = np.amax(COSSENOfiltered)
        #COSSENOfilteredMIN = np.amin(COSSENOfiltered)
    
    # Building ARCTAN image
        ARCTANzerototwopi = np.arctan2(SENO,COSSENO) + np.pi
    #ARCTANzerototwopiMAX = np.amax(ARCTANzerototwopi)
    #ARCTANzerototwopiMIN = np.amin(ARCTANzerototwopi)
    #ARCTANdouble = (ARCTANpitopi + np.pi) / (2 * np.pi) * 255
    #ARCTANdoubleMAX = np.amax(ARCTANdouble)
    #ARCTANdoubleMIN = np.amin(ARCTANdouble)
    
    # Performing phase unwrapping
        imageUNWRAPPED = unwrap_phase(ARCTANzerototwopi)
    #        imageUNWRAPPEDmax = np.amax(imageUNWRAPPED)
    #        imageUNWRAPPEDmin = np.amin(imageUNWRAPPED)
        imageUNWRAPPEDlengthY,imageUNWRAPPEDlengthX = imageUNWRAPPED.shape
    
    # Removing the whole-body deformation
        deltaEDGE = 100
    
        x1,y1 = deltaEDGE,deltaEDGE
        x2,y2 = deltaEDGE,(imageUNWRAPPEDlengthX - deltaEDGE)
        x3,y3 = (imageUNWRAPPEDlengthY - deltaEDGE),(imageUNWRAPPEDlengthX - deltaEDGE)
        
        p1 = np.array([x1, y1, imageUNWRAPPED[y1,x1]])
        p2 = np.array([x2, y2, imageUNWRAPPED[y2,x2]])
        p3 = np.array([x3, y3, imageUNWRAPPED[y3,x3]])
        
        v1 = p3 - p1
        v2 = p2 - p1
        cp = np.cross(v1, v2)
        a, b, c = cp
        d = np.dot(cp, p3)
        
        X = np.arange(0, imageUNWRAPPEDlengthX, 1)
        Y = np.arange(0, imageUNWRAPPEDlengthY, 1)
        X, Y = np.meshgrid(X, Y)
        XX = X.flatten()
        YY = Y.flatten()
        Plano = (d - a*X - b*Y)/c
        
        corrIMAGE = imageUNWRAPPED-Plano
        corrIMAGEmax = np.amax(corrIMAGE)
        corrIMAGEmin = np.amin(corrIMAGE)
        
        #    cv2.imshow('Plan removed',corrIMAGE)
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()
                
        # Save output image
        output = np.uint8((corrIMAGE - corrIMAGEmin) / (corrIMAGEmax-corrIMAGEmin) * 255)
        outputPAD = output
#        outputPAD = cv2.copyMakeBorder(output,0,0,65,65,cv2.BORDER_REFLECT)
    strPART = (listaIMAGENS[i].split("Def"))
    strPART = listaIMAGENS[i]
    strNAME,strEXT = strPART.split(".")
    cv2.imwrite(''.join([DESTcorpos,'/unwRADlev_', strNAME, '_maxRAD', str('{:04.2f}'.format(corrIMAGEmax)), '_minRAD', str('{:04.2f}'.format(corrIMAGEmin)), '.', strEXT]), outputPAD)
#    cv2.imwrite(''.join([DESTcorpos,'/unwRADlev_', strNAME, '_maxRAD', str('{:04.2f}'.format(12)), '_minRAD', str('{:04.2f}'.format(15)), '.', strEXT]), frame)

    
    
    