import os, numpy as np, pandas as pd, cv2, skimage, sys, shutil, multiprocessing as mp, time
import subprocess as sp
from skimage.external import tifffile
import matplotlib.pyplot as plt
import cPickle as pickle
#sudo apt-get install openslide-tools
#pip install openslide-python
#https://openslide.org/api/python/
#https://openslide.org/formats/hamamatsu/
import openslide

def makedir(src):
    if not os.path.exists(src): os.mkdir(src)
    return
def removedir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    return

def listdirfull(src, keyword=False):
    if not keyword: fls = [os.path.join(src, xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx]
    if keyword: fls = [os.path.join(src, xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx and keyword in xx]
    fls.sort()
    return fls

def save_dictionary(dst, dct):
    '''Basically the same as save_kwargs
    '''
    if dst[-2:]!='.p': dst=dst+'.p'

    with open(dst, 'wb') as fl:
        pickle.dump(dct, fl, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_dictionary(pth):
    '''simple function to load dictionary given a pth
    '''
    kwargs = {};
    with open(pth, 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return kwargs


def otsu_filter(vol, otsu_factor=0.8):
    '''
    otsu_factor - scaling of the otsu value, >1 is less stringent, <1 remove more pixels
    '''
    #load
    if type(vol) == str: vol = tifffile.imread(vol)

    #
    v=skimage.filters.threshold_otsu(vol)/float(otsu_factor)
    vol[vol<v]=0
    vol[vol>=v]=1
    return vol

#https://stackoverflow.com/questions/37363755/python-mouse-click-coordinates-as-simply-as-possible?rq=1
import numpy as np
import matplotlib.pyplot as plt

class LineBuilder:
    def __init__(self, line,ax,color):
        self.line = line
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        plt.ioff()
        self.shape_counter = 0
        self.shape = {}
        self.precision = 75
        self.objects = {}
        self.active_object=0

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        #y_x_transform = self.ax.transData.inverted().transform((event.ydata, event.xdata))
        #print y_x_transform, event.ydata, event.xdata
        y_x_transform = (event.xdata, event.ydata) #INTENTIONALLY FLIPPING X AND Y HERE
        if self.counter == 0:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.objects[self.active_object] = [[y_x_transform[0], y_x_transform[1]]] #keeping np convention
        if np.abs(event.xdata-self.xs[0])<=self.precision and np.abs(event.ydata-self.ys[0])<=self.precision and self.counter != 0:
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            self.ax.scatter(self.xs[0],self.ys[0],s=80,color='blue')
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.shape[self.shape_counter] = [self.xs,self.ys]
            self.shape_counter = self.shape_counter + 1
            self.xs = []
            self.ys = []
            self.objects[self.active_object].append([y_x_transform[0], y_x_transform[1]]) #keeping np convention
            self.active_object+=1
            self.counter = 0
        else:
            if self.counter != 0:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1
            self.objects[self.active_object].append([y_x_transform[0], y_x_transform[1]]) #keeping np convention

def create_shape_on_image(data,cmap='jet'):
    def change_shapes(shapes):
        new_shapes = {}
        for i in range(len(shapes)):
            l = len(shapes[i][1])
            new_shapes[i] = np.zeros((l,2),dtype='int')
            for j in range(l):
                new_shapes[i][j,0] = shapes[i][0][j]
                new_shapes[i][j,1] = shapes[i][1][j]
        return new_shapes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Click around each section in order of acquisition,\nclose each shape by clicking close to first point\nclose window when done')
    line = ax.imshow(data)
    ax.set_xlim(0,data[:,:].shape[1])
    ax.set_ylim(0,data[:,:].shape[0])
    linebuilder = LineBuilder(line,ax,'red')
    plt.gca().invert_yaxis()
    new_shapes = change_shapes(linebuilder.shape)
    plt.show(block=True)
    return linebuilder

def mark_edges(src, dst):
    '''#from https://stackoverflow.com/questions/27565939/getting-the-location-of-a-mouse-click-in-matplotlib-using-tkinter
    src = '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output/data/Trtc/M9332_Rabies_2-3 - 2018-07-18 18.04.40-Trtc.tif'
    '''
    src = listdirfull(src)
    for s in src:
        im = tifffile.imread(s)

        #mark objects, return a dictionary of vertices
        plt.ion()
        out = create_shape_on_image(im)
        objects = out.objects

        #save out dict
        dst0= os.path.join(dst, os.path.basename(s)[:-4]+'_user_defined_points.p')

        #optionally check and append
        #if os.path.exists(dst0): objects.update(load_dictionary(dst0))
        save_dictionary(dst0, objects)

    return

def find_pixels_from_contours(pnts_dst, dst0, df, channel, verbose=True):
    '''
    pnts_dst = location of folder containing .p files of human annoations
    dst0 = data folder to save
    df = dataframe
    '''
    sys.stdout.write('Collecting sections, this can take some time...'); sys.stdout.flush()
    dct ={os.path.basename(fl).split('_user_defined')[0].split(channel)[0][:-1]:load_dictionary(fl) for fl in listdirfull(pnts_dst, keyword='.p')}
    p = mp.Pool(mp.cpu_count())
    for basename, ndct in dct.iteritems():
        if bool(ndct):
            tdf = df[df.basename == basename[11:]]
            iterlst = [(basename, dst0, row, level, ndct, verbose) for i,row in tdf.iterrows()]
            p.map(find_pixels_from_contours_helper, iterlst)
    p.terminate()
    if verbose:
            sys.stdout.write('...completed collecting sections\n\n'); sys.stdout.flush()
    return

def find_pixels_from_contours_helper((basename, dst0, row, level, ndct, verbose)):
    '''
    '''

    #make dir
    dst_ch = os.path.join(dst0, row['channel']); makedir(dst_ch)

    #load
    vol = ndpi_to_numpy(os.path.join(row['folder'],row['file']), level=level)

    #segment out sections
    for idx, cnt in ndct.iteritems():
        zero = np.zeros_like(vol)
        cp = np.copy(vol)
        cnt = np.asarray([[int(xx[0]),int(xx[1])] for xx in cnt])
        cv2.fillPoly(zero, pts=[cnt], color=(255,255,255))
        cp[zero==0]=0
        y,x=np.where(cp>0)
        cp=cp[np.min(y):np.max(y), np.min(x):np.max(x)]
        fl = os.path.join(dst_ch, '{}_section_{}.tif'.format(basename, str(idx).zfill(4)))
        tifffile.imsave(fl, cp, compress=1)
        if verbose:
            sys.stdout.write('\n   completed {}: {}'.format(row['channel'], os.path.basename(fl))); sys.stdout.flush()
    return

def pad_first_image(nsrc, other_channel_folders):
    '''function to find biggest image and pad first to accomodate
    '''
    dims = np.asarray([get_dims(fl) for fl in listdirfull(nsrc)])
    ymax, xmax = (dims[:,0].max(), dims[:,1].max())
    first_fl = tifffile.imread(listdirfull(nsrc)[0])
    yd, xd = first_fl.shape
    pad_dims = (int((ymax-yd)/2.0),int((ymax-yd)/2.0)), (int((xmax-xd)/2.0),int((xmax-xd)/2.0))
    first_fl_pad = np.pad(first_fl, pad_dims, mode='constant')
    tifffile.imsave(listdirfull(nsrc)[0], first_fl_pad)
    
    if len(other_channel_folders)>0:
        for ch in other_channel_folders:
            chfl = listdirfull(ch)[0]
            tifffile.imsave(chfl, np.pad(tifffile.imread(chfl), pad_dims, mode='constant'))
    
    return
    
def get_dims(fl):
    with tifffile.TiffFile(fl) as tf:
        dims = tf.asarray().shape
        tf.close()
    return np.asarray(dims)

def read_ndpis(src):
    '''Extract out files
    '''
    df = pd.DataFrame(data = None, columns = ['ndpis', 'NoImages', 'folder', 'basename', 'file', 'channel', 'file_order'])

    for idx, s in enumerate(src):
        with open(s, 'r') as fl:
            lines = fl.readlines()
            fl.close()

        fls = [xx.replace('\r', '').replace('\n', '').split('=')[1] for xx in lines[2:]]

        for fl in fls:
            noimages = lines[1].replace('\r', '').replace('\n', '').split('=')[1]
            flter = fl[fl.rfind('-')+1:fl.rfind('.ndpi')]
            df.loc[len(df)] = [s, noimages, os.path.dirname(s), os.path.basename(s)[:-6], fl, flter, idx]
    return df

def ndpi_to_numpy(src, level=6):
    '''function to load ndpi as np array at level

    level=0 all, increasing number ==less data
    '''
    #read
    fl = openslide.OpenSlide(src)

    #load in ychunks - clunky with uneven breaks
    #chunk_no = 10 #essentially one less chunk than there
    #chunks = np.linspace(0,fl.level_dimensions[1][1], chunk_no, dtype='int')
    #vol = []
    #for i,ii in enumerate(chunks[:-1]):
    #    vol.append(np.array(fl.read_region((0,chunks[i]), level=level, size=tuple((fl.level_dimensions[level][0], chunks[i+1]-chunks[i])))))
    #now append, y is now axis 0
    #vol = np.concatenate(vol, 1)

    vol = fl.read_region((0,0), level=level, size=fl.level_dimensions[level])
    #get rid of alphas
    vol = np.max(np.swapaxes(np.swapaxes(np.array(vol), 0,2),1,2)[:3],0)

    return vol

def convert_data_ndpi_to_tif(dst, df, channel, level=6, verbose=True):
    '''Function to load each ndpi and save out as tiffs
    '''
    #need to preserve order of source
    tdf = df[df.channel == channel]
    assert len(tdf) > 0, 'something went wrong with selecting the alignment channel'
    iterlst = [(row, level, dst, verbose) for i,row in tdf.iterrows()]
    p = mp.Pool(mp.cpu_count())
    p.map(convert_data_ndpi_to_tif_helper, iterlst)
    p.terminate()
    return

def convert_data_ndpi_to_tif_helper((row, level, dst, verbose)):
    '''
    '''
    fl = os.path.join(row['folder'],row['file'])
    out = os.path.join(dst, 'slide_' + str(row['file_order']).zfill(4)+'_'+os.path.basename(fl)[:-4]+'.tif')
    #out = os.path.join(dst, os.path.basename(fl)[:-4]+'_order_' + str(row['file_order']).zfill(4)+'.tif')
    if not os.path.exists(out):
        vol = ndpi_to_numpy(fl, level=level)
        tifffile.imsave(out, vol, compress=1)
        if verbose: print('Completed {}'.format(fl))
    else:
        if verbose: print('{} already exists, skipping'.format(fl))
    return

def elastix_command_line_call(fx, mv, out, parameters, verbose=True):
    '''Wrapper Function to call elastix using the commandline, this can be time consuming

    Inputs
    -------------------
    fx = fixed path (usually Atlas for 'normal' noninverse transforms)
    mv = moving path (usually volume to register for 'normal' noninverse transforms)
    out = folder to save file
    parameters = list of paths to parameter files IN ORDER THEY SHOULD BE APPLIED

    Outputs
    --------------
    ElastixResultFile = '.tif' or '.mhd' result file
    TransformParameterFile = file storing transform parameters

    '''
    e_params=['elastix', '-f', fx, '-m', mv, '-out', out]

    ###adding elastix parameter files to command line call
    for x in range(len(parameters)):
        e_params.append('-p')
        e_params.append(parameters[x])
    if verbose: sys.stdout.write('Elastix Command:\n{}\n...'.format(e_params)); sys.stdout.flush()

    #set paths
    TransformParameterFile = os.path.join(out, 'TransformParameters.{}.txt'.format((len(parameters)-1)))
    ElastixResultFile = os.path.join(out, 'result.{}.tif'.format((len(parameters)-1)))

    #run elastix:
    try:
        if verbose: print ('Running Elastix, this can take some time....\n')
        sp.call(e_params)
        if verbose:
            sys.stdout.write('Past Elastix Commandline Call'); sys.stdout.flush()
    except RuntimeError, e:
        sys.stdout.write('\n***RUNTIME ERROR***: {} Elastix has failed. Most likely the two images are too dissimiliar.\n'.format(e.message)); sys.stdout.flush()
        pass
    if os.path.exists(ElastixResultFile) == True:
        if verbose: sys.stdout.write('Elastix Registration Successfully Completed\n'); sys.stdout.flush()
    #check to see if it was MHD instead
    elif os.path.exists(os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))) == True:
        ElastixResultFile = os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))
        if verbose: sys.stdout.write('Elastix Registration Successfully Completed\n'); sys.stdout.flush()
    else:
        sys.stdout.write('\n***ERROR***Cannot find elastix result file\n: {}'.format(ElastixResultFile)); sys.stdout.flush()
        return

    return ElastixResultFile, TransformParameterFile

def transformix_command_line_call(src, out, tp):
    sp.call(['transformix', '-in', src, '-out', out, '-tp', tp])
    return

def align_sections(nsrc, dst0, parameters, other_channel_folders = False, clean=True):
    '''
    nsrc = list of pre-'cut' files
    dst = location to save
    parameters = list of files in order to register
    clean = T/F if true remove elastix folders
    other_channel_folders = list of folders to apply the same transform to

    nsrc = '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output/data/Trtc'
    dst0 = '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output/data/Trtc_aligned'
    parameters = ['/home/wanglab/LightSheetData/witten-mouse/nanozoomer/align_slices_elastix_parameters.txt']
    clean = True
    other_channel_folders=['/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output/data/YFP', '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output/data/Brighfield', '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output/data/Dapi']
    '''
    fls = listdirfull(nsrc); fls.sort(); makedir(dst0)

    #iterate
    for i, fl in enumerate(fls[:-1]):
        st = time.time()
        if i == len(fls):
            sys.stdout.write('Last z plane completed.\n'); sys.stdout.flush()
            break
        #set up tmp folder
        sys.stdout.write('Aligning section {} with {}...'.format(i+1, i)); sys.stdout.flush()
        tmp = os.path.join(dst0, 'elastix_{}'.format(str(i).zfill(4))); makedir(tmp)
        src_aligned = nsrc+'_aligned'; makedir(src_aligned)

        #get images
        fx_aligned = os.path.join(src_aligned, '{}_aligned_{}.tif'.format(os.path.basename(fls[i])[:-4], str(i).zfill(4)))
        mv_aligned = os.path.join(src_aligned, '{}_aligned_{}.tif'.format(os.path.basename(fls[i+1])[:-4], str(i+1).zfill(4)))
        if i == 0: shutil.copy(fl, fx_aligned)
            
        #register
        outfile, tp = elastix_command_line_call(fx=fx_aligned, mv=fls[i+1], out=tmp, parameters=parameters, verbose=False)

        #move to final
        shutil.copy(outfile, mv_aligned)

        #optionally apply transform to others channels...
        if other_channel_folders:
            sys.stdout.write('applying transforms: '); sys.stdout.flush()
            for ch in other_channel_folders:
                try: #this is done because sometimes people don't take all channels for all ims
                    chdst = ch+'_aligned'; makedir(chdst)
                    sys.stdout.write('{}, '.format(os.path.basename(ch))); sys.stdout.flush()
                    chdst_tmp = os.path.join(chdst, 'tmp'); makedir(chdst_tmp)
                    chfls = listdirfull(ch); chfls.sort()
                    if i == 0:shutil.copy(chfls[i])
                    transformix_command_line_call(nrc = chfls[i+1], out = chdst_tmp, tp=tp)
                    shutil.copy(os.path.join(chdst_tmp, 'result.tif'), os.path.join(chdst, '{}_aligned_{}.tif'.format(os.path.basename(fls[i+1])[:-4], str(i+1).zfill(4))))
                    removedir(chdst_tmp)
                except:
                    sys.stdout.write(' **Missing: {}** '.format(ch)); sys.stdout.flush()

        #clean
        if clean: removedir(tmp)
        sys.stdout.write('...done in {} seconds.\n'.format(int(time.time()-st))); sys.stdout.flush()

def preprocess(src, dst, parameters, level = 6, channel='Trtc', verbose=True):
    '''Function to collect and organize files

    src = list of 'ndpis' files to input files order; is important
    dst = location to save out everything
    parameters = ['/home/wanglab/LightSheetData/witten-mouse/nanozoomer/align_slices_elastix_parameters.txt']
    channel = channel to use for alignment
    '''

    assert np.all([xx[-6:]=='.ndpis' for xx in src])

    #collect and build structure
    df = read_ndpis(src)

    #output
    makedir(dst)
    dst0 = os.path.join(dst, 'data'); makedir(dst0)
    dst1 = os.path.join(dst0, channel+'_registration'); makedir(dst1)
    pnts_dst = os.path.join(dst, 'user_defined_points'); makedir(pnts_dst)

    #Convert registration channel for elastix
    convert_data_ndpi_to_tif(dst1, df, channel, level=level, verbose=verbose)

    #get user to mark points:
    mark_edges(dst1, pnts_dst)
    removedir(dst1)

    #now load and collect sections and save them out
    find_pixels_from_contours(pnts_dst, dst0, df, channel)
    
    #need to expand images to the largest section
    nsrc = os.path.join(dst0, channel)
    other_channel_folders = [xx for xx in listdirfull(dst0) if dst1 not in xx and '_registration' not in xx and '_aligned' not in xx and nsrc not in xx]
    pad_first_image(nsrc, other_channel_folders)
    
    #now register
    align_sections(nsrc, dst0, parameters, other_channel_folders, clean=True)

    return

#%% #MODIFY BELOW
if __name__ == '__main__':
    #note this needs to be run from console for correct gui usage, see README.md for details.
    #Adjust these inputs:

    #list of files in correct order
    src = ['/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-1.ndpis',
           '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-2.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-3 - 2018-07-18 18.04.40.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-4 - 2018-07-18 04.14.19.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-5 - 2018-07-18 02.10.10.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-6 - 2018-07-17 23.50.06.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-7 - 2018-07-17 21.25.57.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-8 - 2018-07-17 19.05.58.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-9_2 - 2018-07-17 17.18.45.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-10.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-11.ndpis',
           #'/home/wanglab/LightSheetData/witten-mouse/nanozoomer/Rabies/M9332_Rabies_2-12 - 2018-07-18 19.53.50.ndpis'
		]

    #location to save data
    dst = '/home/wanglab/LightSheetData/witten-mouse/nanozoomer/output'

    #elastix parameter file to use for alignment
    parameters = ['/home/wanglab/LightSheetData/witten-mouse/nanozoomer/align_slices_elastix_parameters.txt']

    #sectioning depth
    section_depth = 40 #um

    #nanozoomer level of resolution to use; 1 = full resolution - typically this is too large for most things including fiji, 6 = pretty small
    level = 6

    #channel to use for alignment e.g. 'Trtc', 'Brighfield'
    channel= 'Trtc'

    #now close this. Open up a terminal window (if linux, 'ctl + alt + t'). change directories to folder containing this file, then python <name_of_this_file>
    preprocess(src, dst, parameters = parameters, level = level, channel=channel, verbose=True)
