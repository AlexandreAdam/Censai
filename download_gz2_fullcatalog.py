import skimage.io
import pandas as pd
import time
import sys
import os
import urllib


# stuff to make the status pretty.
class Printer():
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def cmdline():
    ''' Controls the command line argument handling for this little program.
    '''

    from optparse import OptionParser

    # read in the cmd line arguments
    USAGE = 'usage:\t %prog [options]\n'
    parser = OptionParser(usage=USAGE)

    # add options
    parser.add_option('--output',
                      dest='output',
                      default='./images',
                      help='Path to save image data')
    parser.add_option('--width',
                      dest='width',
                      default=128,
                      type=int,
                      help='Default width of images')
    parser.add_option('--height',
                      dest='height',
                      default=128,
                      type=int,
                      help='Default height of images')
    parser.add_option('--cat',
                       dest='cat',
                       default='./catalogs/SDSSspecgalsDR14_boada.csv',
                       help='Catalog to get image names from.')
    parser.add_option('--fov',
                        default=15,
                        type=float,
                        )

    (options, args) = parser.parse_args()

    return options, args


def main():

    opt, arg = cmdline()

    # load the data
    df = pd.read_csv(opt.cat)

    width = opt.width
    height = opt.height
    pixelsize = 0.396  # ''/pixel
    scale = opt.fov / pixelsize / width

    # remove trailing slash in output path if it's there.
    opt.output = opt.output.rstrip('\/')

    # total number of images
    n_gals = df.shape[0]

    for row in df.itertuples():
        # the 'scale' parameter is set so that the image will be about 2x the size
        # of the galaxy
        url = ("http://skyserver.sdss.org/dr14/SkyserverWS/ImgCutout/getjpeg"
               "?ra={}"
               "&dec={}"
               "&scale={}"
               "&width={}"
               "&height={}".format(row.ra, row.dec, scale, width, height))
        # if not os.path.isfile('{}/{}.jpg'.format(opt.output, row.objID)):
        if not os.path.isfile('{}/{}.jpg'.format(opt.output, row.dr7objid)): # for gz2
            try:
                img = skimage.io.imread(url)
                # skimage.io.imsave('{}/{}.jpg'.format(opt.output, row.objID),
                skimage.io.imsave('{}/{}.jpg'.format(opt.output, row.dr7objid),
                                  img)
                time.sleep(0.5)
            except urllib.error.HTTPError:
                pass
        current = row.Index / n_gals * 100
        status = "{:.3f}% of {} completed.".format(current, n_gals)
        Printer(status)

    print('')


if __name__ == "__main__":
    main()
