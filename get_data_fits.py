from astropy.io import fits
import pandas as pd
from astropy.nddata import CCDData
import numpy as np

##function used to deal with .fits images

def create_excel_from_fits(fits_file):
    # Opens FITS file
    hdulist = fits.open(fits_file)
    table_data = hdulist[1].data
    # Converts data into panda dataframe
    df = pd.DataFrame(table_data)
    # Converts DataFrame to Excel
    df.to_excel('Image\fits\table_data.xlsx', index=False)


def excel_to_dataframe(name_excel_file):
    # Creates new DataFrame from Excel file
    df_excel = pd.read_excel(name_excel_file)
    indices_to_drop = df_excel.loc[df_excel['CLASS'] != 1].index
    df_excel = df_excel.drop(indices_to_drop)
    return df_excel


def get_alpha_delta(filename):
    str = ""
    pred = ""
    ALPHA_J2000 = ""
    DELTA_J2000 = ""
    count = 0
    while str != "0001_":
        if (pred == "") or (pred != "0" and pred != "1"):
            str = filename[count]
        else:
            str += filename[count]
        pred = filename[count]
        count += 1
    while filename[count] != "_":
        ALPHA_J2000 += filename[count]
        count += 1
    count += 1
    while filename[count] != "_":
        DELTA_J2000 += filename[count]
        count += 1
    return [ALPHA_J2000, DELTA_J2000]


def get_flux_radius_from_dataframe(dataframe_name, ALPHA_J2000, DELTA_J2000):
    flux_radius_alpha = dataframe_name[round(dataframe_name['ALPHA_J2000'], 5) == \
                                       round(float(ALPHA_J2000), 5)].loc[:, ['FLUX_RADIUS']].to_numpy()

    flux_radius_delta = dataframe_name[round(dataframe_name['DELTA_J2000'], 5) == \
                                       round(float(DELTA_J2000), 5)].loc[:, ['FLUX_RADIUS']].to_numpy()
    for fr1 in flux_radius_alpha:
        for fr2 in flux_radius_delta:
            if fr1[0] == fr2[0]:
                FLUX_RADIUS = fr1
                return FLUX_RADIUS


"""TESTS"""

"""
#create_dataframe_from_fits('images/gal_z0p0-0p2.fits')
df_excel = excel_to_dataframe('table_data.xlsx')
a = get_alpha_delta("cutouts/0001_150.12509000_1.61690000_acs_I_mosaic_30mas_sci.fits")
print(get_flux_radius_from_dataframe(df_excel, a[0], a[1]))
"""