import cv2, os, shutil
base_path = "C:/Users/shafi/Desktop/GSF/Demo/"

for filename in os.listdir(base_path):
    file_path = os.path.join(base_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
