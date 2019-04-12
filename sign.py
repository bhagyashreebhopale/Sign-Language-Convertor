import math
import cv2
from Levenshtein import *
from PIL import Image
import Levenshtein
import imutils
import pyttsx3
engine = pyttsx3.init()


class BWImageCompare(object):
    """Compares two images (b/w)."""

    _pixel = 255
    _colour = False

    def __init__(self, imga, imgb, maxsize=64):
        """Save a copy of the image objects."""

        sizea, sizeb = imga.size, imgb.size

        newx = min(sizea[0], sizeb[0], maxsize)
        newy = min(sizea[1], sizeb[1], maxsize)

        # Rescale to a common size:
        imga = imga.resize((newx, newy), Image.BICUBIC)
        imgb = imgb.resize((newx, newy), Image.BICUBIC)

        if not self._colour:
            # Store the images in B/W Int format
            imga = imga.convert('I')
            imgb = imgb.convert('I')

        self._imga = imga
        self._imgb = imgb

        # Store the common image size
        self.x, self.y = newx, newy

    def _img_int(self, img):
        """Convert an image to a list of pixels."""

        x, y = img.size

        for i in xrange(x):
            for j in xrange(y):
                yield img.getpixel((i, j))

    @property
    def imga_int(self):
        """Return a tuple representing the first image."""

        if not hasattr(self, '_imga_int'):
            self._imga_int = tuple(self._img_int(self._imga))

        return self._imga_int

    @property
    def imgb_int(self):
        """Return a tuple representing the second image."""

        if not hasattr(self, '_imgb_int'):
            self._imgb_int = tuple(self._img_int(self._imgb))

        return self._imgb_int

    @property
    def mse(self):
        """Return the mean square error between the two images."""

        if not hasattr(self, '_mse'):
            tmp = sum((a-b)**2 for a, b in zip(self.imga_int, self.imgb_int))
            self._mse = float(tmp) / self.x / self.y

        return self._mse

    @property
    def psnr(self):
        """Calculate the peak signal-to-noise ratio."""

        if not hasattr(self, '_psnr'):
            self._psnr = 20 * math.log(self._pixel / math.sqrt(self.mse), 10)

        return self._psnr

    @property
    def nrmsd(self):
        """Calculate the normalized root mean square deviation."""

        if not hasattr(self, '_nrmsd'):
            self._nrmsd = math.sqrt(self.mse) / self._pixel

        return self._nrmsd

    @property
    def levenshtein(self):
        """Calculate the Levenshtein distance."""

        if not hasattr(self, '_lv'):
            stra = ''.join((chr(x) for x in self.imga_int))
            strb = ''.join((chr(x) for x in self.imgb_int))

            lv = distance(stra, strb)

            self._lv = float(lv) / self.x / self.y

        return self._lv
def distance(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])


class ImageCompare(BWImageCompare):


    _pixel = 255 ** 3
    _colour = True

    def _img_int(self, img):
        """Convert an image to a list of pixels."""

        x, y = img.size

        for i in range(x):
            for j in range(y):
                pixel = img.getpixel((i, j))
                yield pixel[0] | (pixel[1]<<8) | (pixel[2]<<16)

    @property
    def levenshtein(self):
        """Calculate the Levenshtein distance."""

        if not hasattr(self, '_lv'):
            stra_r = ''.join((chr(x>>16) for x in self.imga_int))
            strb_r = ''.join((chr(x>>16) for x in self.imgb_int))
            lv_r = distance(stra_r, strb_r)

            stra_g = ''.join((chr((x>>8)&0xff) for x in self.imga_int))
            strb_g = ''.join((chr((x>>8)&0xff) for x in self.imgb_int))
            lv_g = distance(stra_g, strb_g)

            stra_b = ''.join((chr(x&0xff) for x in self.imga_int))
            strb_b = ''.join((chr(x&0xff) for x in self.imgb_int))
            lv_b = distance(stra_b, strb_b)

            self._lv = (lv_r + lv_g + lv_b) / 3. / self.x / self.y

        return self._lv


class FuzzyImageCompare(object):
    """Compares two images based on the previous comparison values."""

    def __init__(self, imga, imgb, lb=1, tol=15):
        """Store the images in the instance."""

        self._imga, self._imgb, self._lb, self._tol = imga, imgb, lb, tol

    def compare(self):
        """Run all the comparisons."""

        if hasattr(self, '_compare'):
            return self._compare

        lb, i = self._lb, 2

        diffs = {
            'levenshtein': [],
            'nrmsd': [],
            'psnr': [],
        }

        stop = {
            'levenshtein': False,
            'nrmsd': False,
            'psnr': False,
        }

        while not all(stop.values()):
            cmp = ImageCompare(self._imga, self._imgb, i)

            diff = diffs['levenshtein']
            if len(diff) >= lb+2 and \
                abs(diff[-1] - diff[-lb-1]) <= abs(diff[-lb-1] - diff[-lb-2]):
                stop['levenshtein'] = True
            else:
                diff.append(cmp.levenshtein)

            diff = diffs['nrmsd']
            if len(diff) >= lb+2 and \
                abs(diff[-1] - diff[-lb-1]) <= abs(diff[-lb-1] - diff[-lb-2]):
                stop['nrmsd'] = True
            else:
                diff.append(cmp.nrmsd)

            diff = diffs['psnr']
            if len(diff) >= lb+2 and \
                abs(diff[-1] - diff[-lb-1]) <= abs(diff[-lb-1] - diff[-lb-2]):
                stop['psnr'] = True
            else:
                try:
                    diff.append(cmp.psnr)
                except ZeroDivisionError:
                    diff.append(-1)  # to indicate that the images are identical

            i *= 2

        self._compare = {
            'levenshtein': 100 - diffs['levenshtein'][-1] * 100,
            'nrmsd': 100 - diffs['nrmsd'][-1] * 100,
            'psnr': diffs['psnr'][-1] == -1 and 100.0 or diffs['psnr'][-1],
        }

        return self._compare

    def similarity(self):
        """Try to calculate the image similarity."""

        cmp = self.compare()

        lnrmsd = (cmp['levenshtein'] + cmp['nrmsd']) / 2
        return lnrmsd
        return min(lnrmsd * cmp['psnr'] / self._tol, 100.0)  # TODO: fix psnr!


if __name__ == '__main__':

    #import sys
 
    #if len(sys.argv) < 3:
     #   print ('usage: %s image-file-1.jpg image-file-2.jpg ...' % sys.argv[0])
      #  sys.exit()

    #tot = len(sys.argv) - 1
    #tot = (tot ** 2 - tot) / 2

    #print('Comparing %d images:' % tot)
    t=True;
    camera = cv2.VideoCapture(0) 
    top, right, bottom, left = 10, 350, 325, 890
    num_frames = 0
    while(t):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[102:298, 427:623]
        cv2.imwrite("%d.jpg" % 1, roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.rectangle(clone, (425, 100), (625, 300), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("e"):
            break
        if num_frames>50:
            break

    #edges = cv2.Canny(image,50,100)
        #cv2.imwrite("edge%d.jpg"%count, edges)
    #cv2.imshow("img", edges)
    images = {}
    images[0] =  Image.open(("%d.jpg"%1))

    img=1;
   # ch='A'
    #while ch<'V':
     #   images[img] = Image.open((chr(ord(ch)) + ".jpg"))
      #  ch=chr(ord(ch)+1)
           # img+=1
    ch=1;
    while ch<9:
       images[img] = Image.open("00%d.jpg"%ch)
       ch+=1
       img+=1
    
    #for img in sys.argv[1:]:
        #images[img] = Image.open(img

    results, i = {}, 1
    for namea, imga in images.items():
        for nameb, imgb in images.items():
            if namea == nameb or (nameb, namea) in results:
                continue

          #  print (' * %2d / %2d:' % (i, tot),)
            print (namea, nameb, '...',)

            cmp = FuzzyImageCompare(imga, imgb)
            sim = cmp.similarity()
            results[(namea, nameb)] = sim

            print ('%.2f %%' % sim)

            i += 1
        break

    res = max(results.values())
    #k=0
    imgs = [k for k, v in results.items() if v == res][0]
    ans = ["","A","B","C","D","E","F","G","H","I"]
    c = imgs[1]
    engine.say(ans[c])
    engine.setProperty('rate',120)
    engine.setProperty('volume', 0.9)
    engine.runAndWait()
    print(ans[c])
    print('Most similar images: %s %s (%.2f %%)' % (imgs[0], imgs[1], res))
    print('Most similar images: %s ' % ( imgs[1]))
