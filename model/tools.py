import cv2
import os

def resize_images(width, height, image_list, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    images = open(image_list).readlines()

    print 'resize {} samples...'.format(len(images))
    for i, img_path in enumerate(images):
        print '[{}] {}'.format(i, img_path)
        img_path = img_path.strip()
        img_id = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(out_dir, img_id)
        cv2.imwrite(out_path, img)

    print 'Done!'

def main():
    import sys
    image_list = sys.argv[1]
    out_dir = sys.argv[2]
    resize_images(224, 224, image_list, out_dir)

if __name__ == '__main__':
    main()

