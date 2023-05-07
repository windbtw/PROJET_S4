import scipy.ndimage as ndi
from astropy.visualization import simple_norm
from astropy.convolution import convolve
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from photutils.segmentation import detect_threshold
from photutils.segmentation import detect_sources
import statmorph
from galaxy_functions import *
from get_data_fits import *

##all the code necessary to get the data we want from an  image, with segmentation and statmorph

LOG_A = 1.0  # 10000 initially but doesn't work well with fits format

def get_galaxy_stat(file_path, excel_dataframe_name=None, plot=False, image_format="fits", zoom_on_image=False):
    """
    :param file_path : image file path
    :param excel_dataframe_name : only necessary for fits images which aren't properly zoomed in,
    allows to automatically find the correct FLUX_RADIUS
    :param plot : set to 'True' if you want to see the image decomposition
    :param image_format : either image_format = fits or png, default setting is 'fits'. jpeg images can be labeled as 'png'
    :param zoom_on_image : True or False, False by default
    :return: : CAS, Gini, M20
    """

    """Converting image to galaxy model"""

    image = None
    nx, ny = None, None

    if image_format == "jpeg":
        image = load_image(file_path)
        ny, nx = image.shape[0], image.shape[1]
        image = convert_to_image(image)

    elif image_format == "fits":

        if zoom_on_image:
            if excel_dataframe_name == None:
                print("Error : you must give the name of the dataframe linked to the fits image")
                return False

            image_file = get_pkg_data_filename(file_path)
            image = fits.getdata(image_file, ext=0)
            df = excel_to_dataframe(excel_dataframe_name)
            alpha_delta = get_alpha_delta(file_path)
            ALPHA_J2000 = alpha_delta[0]
            DELTA_J2000 = alpha_delta[1]
            FLUX_RADIUS = get_flux_radius_from_dataframe(df, ALPHA_J2000, DELTA_J2000)[0]
            image = zoom(image, round(FLUX_RADIUS))
            ny, nx = image.shape[0], image.shape[1]

        else:
            image_file = get_pkg_data_filename(file_path)
            image = fits.getdata(image_file, ext=0)
            ny, nx = image.shape[0], image.shape[1]


    if plot:
        plot_galaxy(image)

    """Convolving with a PSF"""

    size = 20  # on each side from the center
    y, x = np.mgrid[-size:size+1, -size:size+1]
    sigma_psf = 2.0
    psf = np.exp(-(x**2 + y**2)/(2.0*sigma_psf**2))
    psf /= np.sum(psf)
    if plot:
        plt.imshow(psf, origin='lower', cmap='gray')
        plt.show()
    image = convolve(image, psf)
    if plot:
        plt.imshow(image, cmap='gray', origin='lower',
            norm=simple_norm(image, stretch='log', log_a=LOG_A))
        plt.show()

    """Adding noise"""

    np.random.seed(1)
    snp = 100.0
    image += (1.0 / snp) * np.random.standard_normal(size=(ny, nx))      # 1 initially, 2000 can also work
    if plot:
        plt.imshow(image, cmap='gray', origin='lower',
            norm=simple_norm(image, stretch='log', log_a=LOG_A))
        plt.show()

    """Gain and weight maps"""

    gain = 10000.0

    """Creating a segmentation map"""

    threshold = detect_threshold(image, 1.5)
    npixels = 5  # minimum number of connected pixels
    segm = detect_sources(image, threshold, npixels)
    label = np.argmax(segm.areas) + 1  # Keep only the largest segment
    segmap = segm.data == label
    if plot:
        plt.imshow(segmap, origin='lower', cmap='gray')
        plt.show()
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = np.int64(segmap_float > 0.5)
    if plot:
        plt.imshow(segmap, origin='lower', cmap='gray')
        plt.show()

    """Running Statmorph"""

    source_morphs = statmorph.source_morphology(
        image, segmap, gain=gain, psf=psf)
    morph = source_morphs[0]

    print('M20 =', morph.m20)
    print('Gini =', morph.gini)
    print('C =', morph.concentration)
    print('A =', morph.asymmetry)
    print('S =', morph.smoothness)
    return(abs(morph.concentration), abs(morph.asymmetry), abs(morph.smoothness), abs(morph.m20), abs(morph.gini))