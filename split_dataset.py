import os
import imageio
import shutil
import patoolib
import imageio

if __name__ == "__main__":


    if not os.path.exists("UCF11_updated_mpg.rar"):
        u = "http://crcv.ucf.edu/data/UCF11_updated_mpg.rar"
        urllib.request.urlretrieve (u, "UCF11_updated_mpg.rar")

    if not os.path.exists("UCF11_updated_mpg"):
        patoolib.extract_archive("UCF11_updated_mpg.rar", outdir='.')

    root='UCF11_updated_mpg'
    classes=sorted(os.listdir(root))

    path = 'UCF11_split'
    train_path = path+'/train'
    test_path = path+'/test'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for c in classes:
        subclasses = sorted(os.listdir(root+'/'+c))
        n=len(subclasses)-2
        if not os.path.exists(train_path+'/'+c):
            os.makedirs(train_path+'/'+c)
        if not os.path.exists(test_path+'/'+c):
            os.makedirs(test_path+'/'+c)

        for i,subclass in enumerate(subclasses):
            if subclass=='.DS_Store' or subclass=='Annotation':
                continue
            if i<(0.8*n):
                videos = sorted(os.listdir(root+'/'+c+'/'+subclass))
                for video in videos:
                    src_dir = root+'/'+c+'/'+subclass+'/'+video
                    dst_dir = train_path+'/'+c+'/'+video
                    shutil.copy(src_dir,dst_dir)
            else:
                videos = sorted(os.listdir(root+'/'+c+'/'+subclass))
                for video in videos:
                    src_dir = root+'/'+c+'/'+subclass+'/'+video
                    dst_dir = test_path+'/'+c+'/'+video
                    shutil.copy(src_dir,dst_dir)

