import cv2, os, shutil
base_path = "C:/Users/shafi/Desktop/Demo/"       
new_path = "C:/Users/shafi/Desktop/Demo/"
d=1
for infile in os.listdir(base_path):
    print ("file : " + infile)
    read = cv2.imread(base_path + infile)
    #outfile = infile.split('.')[0] + '.jpg'
    outfile = '%d.jpg'%d
    cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    d+=1
