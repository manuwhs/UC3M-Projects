import matplotlib.pyplot as plt
import img_util as imgu
import numpy as np
import PIL

pil_im = PIL.Image.open("foo.pgm") # Read image and created image object
pil_im = pil_im.convert('L')  # Convert image to grayScale
pil_arr_im = np.array(pil_im);  # Convert to simple matrix 


SHOW_IMAGE = 0

if (SHOW_IMAGE == 1):
    plt.figure()
    
    # plot the image
    plt.imshow(pil_arr_im, plt.cm.gray)
    
    # some points
    x = [41,12,58,23]
    y = [14,35,21,48]
    # plot the points with red star-markers
    plt.plot(x,y,'r*')
    
    # line plot connecting the first two points
    plt.plot(x[:2],y[:2])
    # add title and show the plot
    plt.title('Plotting: "face1.jpg"')
    
    plt.show()
    
SHOW_HISTOGRAM = 0

if (SHOW_HISTOGRAM == 1):
    nbr_bins = 255;
    imhist,bins = np.histogram(pil_arr_im.flatten(),nbr_bins,normed=True)

    plt.figure()
    plt.bar(bins[:255],imhist,width = 1 ,color='c',align='center')
    plt.title('Linear Discriminant Analysis accuracy')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.show()
    
SHOW_EQ_HISTOGRAM = 0
if (SHOW_EQ_HISTOGRAM == 1):
    nbr_bins = 255;
    pil_arr_im_eq = imgu.histeq(pil_arr_im);
    imhist,bins = np.histogram(pil_arr_im_eq.flatten(),nbr_bins,normed=True)

    plt.figure()
    plt.bar(bins[:255],imhist,width = 1 ,color='c',align='center')
    plt.title('Linear Discriminant Analysis accuracy')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.show()
    
    plt.figure()
    # plot the image
    plt.imshow(pil_arr_im, plt.cm.gray)
    
PLOT_ALL = 1
if (PLOT_ALL == 1):
    
    nbr_bins = 255;
    imhist,bins = np.histogram(pil_arr_im.flatten(),nbr_bins,normed=True)
    pil_arr_im_eq = imgu.histeq(pil_arr_im);
    imhist_eq,bins = np.histogram(pil_arr_im_eq.flatten(),nbr_bins,normed=True)
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(pil_arr_im, plt.cm.gray)
    ax2.imshow(pil_arr_im_eq, plt.cm.gray)
    ax3.bar(bins[:255],imhist,width = 1 ,color='c',align='center')
    ax4.bar(bins[:255],imhist_eq,width = 1 ,color='c',align='center')
    
    
"""
The histogram only tells us how many pixels of each "value" there are in the image,
it does not give any spatial relationship
"""

## READ PGM image and show it !!
image = imgu.read_pgm("foo.pgm", byteorder='<')
plt.imshow(image, plt.cm.gray)
plt.show()
