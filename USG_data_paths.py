import os
spectra_types = {'Artificial': 'ChapterA_ArtificialMaterials',
'Coatings':' ChapterC_Coatings',
'Liquids':'ChapterL_Liquids',
'Minerals': 'ChapterM_Minerals',
'Organic': 'ChapterO_OrganicCompounds',
'Soils': 'ChapterS_SoilsAndMixtures',
'Vegetation': 'ChapterV_Vegetation',
'Test_set':'Test'}

sensor_type = {'splib07a': 'ASCIIdata_splib07a',
'splib07b': 'ASCIIdata_splib07b',
'ASD': 'ASCIIdata_splib07b_cvASD',
'AVIRIS1995': 'ASCIIdata_splib07b_cvAVIRISc1995',
'AVIRIS1996': 'ASCIIdata_splib07b_cvAVIRISc1996',
'AVIRIS1997': 'ASCIIdata_splib07b_cvAVIRISc1997',
'AVIRIS1998': 'ASCIIdata_splib07b_cvAVIRISc1998',
'AVIRIS1999': 'ASCIIdata_splib07b_cvAVIRISc1999',
'AVIRIS2000': 'ASCIIdata_splib07b_cvAVIRISc2000',
'AVIRIS2001': 'ASCIIdata_splib07b_cvAVIRISc2001',
'AVIRIS2005': 'ASCIIdata_splib07b_cvAVIRISc2005',
'AVIRIS2006': 'ASCIIdata_splib07b_cvAVIRISc2006',
'AVIRIS2009': 'ASCIIdata_splib07b_cvAVIRISc2009',
'AVIRIS2010': 'ASCIIdata_splib07b_cvAVIRISc2010',
'AVIRIS2011': 'ASCIIdata_splib07b_cvAVIRISc2011',
'AVIRIS2012': 'ASCIIdata_splib07b_cvAVIRISc2012',
'AVIRIS2013': 'ASCIIdata_splib07b_cvAVIRISc2013',
'AVIRIS2014': 'ASCIIdata_splib07b_cvAVIRISc2014',
'CRISMGlobal': 'ASCIIdata_splib07b_cvCRISM-global',
'CRISMjMTR3': 'ASCIIdata_splib07b_cvCRISMjMTR3',
'HYMAP2007': 'ASCIIdata_splib07b_cvHYMAP2007',
'HYMAP2014': 'ASCIIdata_splib07b_cvHYMAP2014',
'HYPERION': 'ASCIIdata_splib07b_cvHYPERION',
'M3target': 'ASCIIdata_splib07b_cvM3-target',
'VIMS': 'ASCIIdata_splib07b_cvVIMS',
'ASTER': 'ASCIIdata_splib07b_rsASTER',
'Landsat8': 'ASCIIdata_splib07b_rsLandsat8',
'Sentinel12': 'ASCIIdata_splib07b_rsSentinel2',
'WorldView3': 'ASCIIdata_splib07b_rsWorldView3'}

dataPath = os.path.join('C:\\','hyper_data')
dataPath_HSI = os.path.join(dataPath,'hyperspectral_images')

two_member = [[0,1],[1,2], [2,3], [3,4], [4,0]]
three_member= [[0,1,2], [1,2,3], [2,3,4], [3,4,0], [4,0,1]]
four_member = [[0,1,2,3], [1,2,3,4], [2,3,4,0], [3,4,0,1], [4,0,1,2]]
