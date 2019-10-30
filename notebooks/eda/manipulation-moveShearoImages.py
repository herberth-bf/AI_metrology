import os
import numpy as np
import glob
import shutil
from PIL import Image

def copy_move(src_dir, dst_dir):
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.bmp")):
        shutil.copy(jpgfile, dst_dir)
    
# train
trainEnergies =  np.array([])
base_train  = r"D:\MachineLearningInOpticsExtrapolation\NoDecomposition\TrainSet"
dest_train = r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TrainSet"
folders = os.listdir(base_train)
# Trocar todos os nomes dos arquivos em ordem crescente
aux=0
for folder in folders:
    fullFolder = os.path.join(base_train, folder)
    files = os.listdir(fullFolder)
    for file in files:
        os.rename(os.path.join(base_train, folder, file), os.path.join(base_train, folder, str(aux)+".bmp", ))
        aux+=1       
for folder in folders: # subfolders
    files = os.listdir(os.path.join(base_train, folder))
    targets = np.ones(len(files))*float(folder)
    trainEnergies = np.concatenate([trainEnergies, targets])
    copy_move(os.path.join(base_train, folder), dest_train)
    
np.savetxt(os.path.join(dest_train, "trainEnergies.txt"), trainEnergies)


# test
testEnergies = np.array([])
base_test  = r"D:\MachineLearningInOpticsExtrapolation\NoDecomposition\TestSet"
dest_test = r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\TestSet"
folders = os.listdir(base_test)
# trocar nome dos arquivos
aux=0
for folder in folders:
    fullFolder = os.path.join(base_test, folder)
    files = os.listdir(fullFolder)
    for file in files:
        os.rename(os.path.join(base_test, folder, file), os.path.join(base_test, folder, str(aux)+".bmp", ))
        aux+=1 
        
for folder in folders: # subfolders
    files = os.listdir(os.path.join(base_test, folder))
    targets = np.ones(len(files))*float(folder)
    testEnergies = np.concatenate([testEnergies, targets])
    copy_move(os.path.join(base_test, folder), dest_test)
np.savetxt(os.path.join(dest_test, "testEnergies.txt"), testEnergies)

# val
valEnergies = np.array([])
base_val  = r"D:\MachineLearningInOpticsExtrapolation\NoDecomposition\ValSet"
dest_val = r"D:\MachineLearningInOpticsExtrapolation\NoDecompositionRegression\ValSet"
folders = os.listdir(base_val)
# trocar nome dos arquivos
aux=0
for folder in folders:
    fullFolder = os.path.join(base_val, folder)
    files = os.listdir(fullFolder)
    for file in files:
        os.rename(os.path.join(base_val, folder, file), os.path.join(base_val, folder, str(aux)+".bmp", ))
        aux+=1 
for folder in folders: # subfolders
    files = os.listdir(os.path.join(base_val, folder))
    targets = np.ones(len(files))*float(folder)
    valEnergies = np.concatenate([valEnergies, targets])
    copy_move(os.path.join(base_val, folder), dest_val)
np.savetxt(os.path.join(dest_val, "valEnergies.txt"), valEnergies)
