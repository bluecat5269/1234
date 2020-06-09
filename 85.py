from skimage.measure import compare_ssim
import cv2

class CompareImage():

    def compare_image(self, path_image1, path_image2):

        imageA = cv2.imread(path_image1)
        imageA= cv2.resize(imageA, (400, 400),interpolation=cv2.INTER_LANCZOS4)
        imageB = cv2.imread(path_image2)
        imageB= cv2.resize(imageB, (400, 400),interpolation=cv2.INTER_LANCZOS4)

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        print("SSIM: {}".format(score))
        return score
    
compare_image = CompareImage()
compare_image.compare_image("test7.jpg", "test4.jpg")