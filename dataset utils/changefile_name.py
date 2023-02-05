import os

if __name__ == '__main__':
    for num in range(1, 6):
        path = './picture/fry egg/' + str(num)
        image_dir = sorted(os.listdir(path))
        for i, file in enumerate(image_dir):
            file1 = str(num)+"_" + file
            os.rename(path+'/'+file, path+'/'+file1)
