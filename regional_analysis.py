import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import seaborn as sns
from sklearn.decomposition import PCA
sys.path.append('../')
from utils import *
from processing import *
import cv2 as cv
from image_registration import *
from skimage import color
import scipy as sp
import h5py
import SimpleITK as sitk
from skimage.transform import rescale,resize
from pystackreg import StackReg
from PIL import Image
import nibabel as nib
import ipywidgets as widgets
from ipywidgets import Box, IntSlider
import umap
import image_registration
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.preprocessing import OneHotEncoder
import shap


def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(int)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(int)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(int)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(int)
    return np.copy(np.pad(img, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant'))

def IonImg_transform(data, spec_idx_transform, background):
    if background:
        ion_img_transform = np.zeros(spec_idx_transform.size)
    else:
        ion_img_transform = np.empty(spec_idx_transform.size)
        ion_img_transform[:] = np.nan
    img_shape = spec_idx_transform.shape
    spec_idx_transform_flatten = spec_idx_transform.flatten()
    ion_img_transform[spec_idx_transform_flatten!=0] = data[spec_idx_transform_flatten[spec_idx_transform_flatten!=0]-1]
    ion_img_transform = ion_img_transform.reshape(img_shape)
    return ion_img_transform


def main(do_umap=True,model_train=True):

    with open('./processed_data/Coronal3D_UMAP_datainfo_decoded.pkl', 'rb') as fp:
        UMAP_datainfo = pickle.load(fp)
        
    slice_order = ['slide1_2_R00','slide1_2_R01','slide1_2_R02','slide1_2_R03','slide1_2_R04','slide1_2_R05','slide1_2_R06',
                'slide1_2_R07','slide1_2_R08','slide3_5_R00','slide3_5_R10','slide3_5_R01','slide3_5_R02','slide3_5_R03',
                'slide3_5_R04','slide4_6_R00','slide4_6_R11','slide4_6_R01','slide4_6_R02','slide4_6_R03','slide4_6_R04',
                'slide3_5_R05']
    
    ###load MRI atlas data
    file_dir = './'
    epi_labels = nib.load(file_dir+'WHS_SD_rat_atlas_v2.nii.gz')
    epi_t2 = nib.load(file_dir+'WHS_SD_rat_T2star_v1.01.nii.gz')
    epi_fa = nib.load(file_dir+'WHS_SD_rat_FA_color_v1.01.nii.gz')
    epi_labels_data = epi_labels.get_fdata()
    epi_t2_data = epi_t2.get_fdata()

    mri_slice_num = [645,640,635,630,625,613,608,565,561,557,553,549,545,541,536,531,525,520,515,510,505,500,365,360,355,350,330,325,320,310,305,295,290,280,275,270,265]

    mri_imgs = []
    label_imgs = []
    for i in range(len(mri_slice_num)):
        slice_no = mri_slice_num[i]

        label_slice = epi_labels_data[:, slice_no, :].copy()
        mri_slice = epi_t2_data[:, slice_no, :].copy()
        label_slice_mask = label_slice.copy()
        label_slice_mask[label_slice_mask>0] = 1

        mri_slice_masked = mri_slice*label_slice_mask
        #mri_slice_masked = (mri_slice_masked - np.min(mri_slice_masked)) / (np.max(mri_slice_masked) - np.min(mri_slice_masked))
        mri_slice_masked = mri_slice_masked/mri_slice_masked.max()
        #mri_slice_masked[label_slice_mask==0]+=1
        mri_slice_masked = np.flip(mri_slice_masked.T)
        mri_slice_label = np.flip(label_slice.T)

        mri_imgs.append(mri_slice_masked)
        label_imgs.append(mri_slice_label)
    ###
    
    ### load processed 3D MSI data
    spec_idx_transformed = loadmat('./image_register/Matlab Registration/coronal_registration/spec_idx_imgs_transformed.mat')['spec_idx_imgs_transformed']

    atlas_region = {'basal forebrain':82,'corpus collusum':67,'cortex':92,'corticofugal pathway':1,'hippocampal region':[95,96,97,98],'hypothalamus':48,'striatum':30,
                'thalamus':39,'brainstem':47,'superior colliculus':[50,55],'septal region':40}

    file_dir = './processed_data/coronal3D_propagated_decoded_new.h5'
    reg = image_registration.Image(data_dir=file_dir,registration_dir='../image_transform',group_order=slice_order)
    reg.load_transform('./image_register/Matlab Registration/coronal_registration')

    print('loading processed 3D data...')
    reg.load_data(True)
    ###
    
    ### define brain regions
    colormaps = {'corpus collusum':(1,124/255,0,1), 'cortex':(252/255,245/255,0,1), 'corticofugal pathway':(0,1,246/255,1),
        'hippocampal region':(132/255,1,0,1), 'hypothalamus':(0,140/255,1,1), 'thalamus':(0,16/255,1,1),
            'brainstem':(114/255,0,1,1),'superior colliculus':(238/255,0,1,1),'basal forebrain':(1,0,0,1), 
                'septal region':(0,1,116/255,1), 'striatum':(8/255,1,0,1)}
    ###
    
    
    for slice_no in range(len(slice_order)):
    
        ### performing analysis by extracting intensity profiles for each brain for individual tissue section.
        intens_mtx = reg.data[slice_order[slice_no]]['intens_mtx']

        regions = []
        data = []
        region_img_idx = []
        for key in atlas_region.keys():
            region_idx = np.array(sorted(list(set(spec_idx_transformed[:,:,slice_no].T[np.isin(label_imgs[slice_no],atlas_region[key])].astype(int)-1))))
            if region_idx.shape[0]>int(0.01*intens_mtx.shape[0]):
                region_data = intens_mtx[region_idx[region_idx>=0]]
                data.append(region_data)
                regions+=[key]*region_data.shape[0]
                region_img_idx.append(region_idx[region_idx>=0])

        region_img_idx = np.concatenate(region_img_idx)

        region_data = np.concatenate(data)
        region_data_df = pd.DataFrame(region_data)
        mzs = reg.mzs.round(4)
        region_data_df.columns = mzs
        region_data_df['region'] = regions

        ### perform UMAP analysis on the regional intensity profiles
        if do_umap:
            print('performing UMAP on slice number {} out of {}'.format(slice_no, len(slice_order)))
            region_umap = umap.UMAP(n_components=2,n_neighbors=30,
                            min_dist=0.4,metric='cosine',random_state=19).fit_transform(region_data)


            umap_spec_df = pd.DataFrame({'umap1':region_umap[:,0],'umap2':region_umap[:,1],
                                            'regions':region_data_df['region']})
            plt.figure(figsize=(4,4))
            region_show = atlas_region.keys()
            #region_show = ['molecular layer cerebellum','granule cell layer cerebellum']
            sns.scatterplot(x='umap1',y='umap2',hue='regions',data=umap_spec_df[umap_spec_df['regions'].isin(region_show)],
                            palette=[colormaps[key] for key in umap_spec_df['regions'].unique()],alpha=0.8,edgecolor=None,s=1,rasterized=True)
            plt.legend(frameon=False,fontsize=12,ncol=1,loc='center left',bbox_to_anchor=(1, 0.5))

            plt.savefig('./figures/2d_coronal_spec_umap_slice_{}.pdf'.format(slice_no))
            plt.close()
            umap_spec_df.to_pickle('./processed_data/2d_coronal_spec_umap_slice_{}.pkl'.format(slice_no))
        ###
        
        ### train classifier to predict brain regions given intensity profiles and explain through SHAP values
        if model_train:

            print('training models on slice number {} out of {}'.format(slice_no, len(slice_order)))
            model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=4, n_estimators=100, n_jobs=8, verbose_eval=True)
            enc = OneHotEncoder(handle_unknown='ignore')
            region_onehot = enc.fit_transform(region_data_df['region'].values.reshape(-1,1)).toarray()
            region_encoded = np.argmax(region_onehot,axis=1)

            X_train, X_valid, y_train, y_valid = train_test_split(region_data, region_encoded, test_size=0.1, random_state=7)
            model_dir = './regional_models/model_slice_{}.json'.format(slice_no)

            if os.path.isfile(model_dir):
                model.load_model(model_dir)
            else:
                model.fit(X_train, y_train)
                model.save_model(model_dir)

            pred = model.predict_proba(X_valid)
            pred_label = np.argmax(pred,axis=1)

            cf_matrix = confusion_matrix(y_valid,pred_label)
            plot_confusion_matrix(cf_matrix, classes=enc.get_feature_names())
            plt.savefig('./figures/confuse_mtx_slice_{}.pdf'.format(slice_no))
            plt.close()

            explainer = shap.TreeExplainer(model)
            SHAP_values = []
            for i in tqdm(range(0,region_data.shape[0],10)):
                shap_values = explainer.shap_values(region_data[i:i+10,:])
                SHAP_values.append(shap_values)
            
            SHAP_values_all = []
            for i in range(len(region_data_df['region'].unique())):
                SHAP_values_all.append(np.concatenate([shap_list[i] for shap_list in SHAP_values]))
            SHAP_values_all = np.stack(SHAP_values_all)
            np.save('./processed_data/shap_all_slice_{}.npy'.format(slice_no),SHAP_values_all)

            mean_shap = np.abs(SHAP_values_all).mean(1)
            mean_shap_df = pd.DataFrame(mean_shap.T,columns=enc.categories_[0])
            
            mean_shap_df.to_pickle('./processed_data/mean_shap_slice_{}.pkl'.format(slice_no))
            rank_shap = np.argsort(mean_shap.mean(0))
            
            ax = sns.clustermap(mean_shap_df.iloc[rank_shap[::-1][:20]],row_cluster=False,cmap="mako",figsize=(3,5),z_score=0)
            ax.ax_heatmap.axes.set_yticks(np.arange(0,20),rotation=90)
            ax.ax_heatmap.axes.set_yticklabels(region_data_df.columns[rank_shap[::-1]][:20],rotation=0)
            ax.ax_heatmap.axes.yaxis.tick_right()
            ax.cax.set_visible(False)
            plt.savefig('./figures/meanshap_top20_slice_{}.pdf'.format(slice_no))
            plt.close()


            print('obtaining shap umap image...')
            explainer = shap.TreeExplainer(model)
            SHAP_values = []
            for i in tqdm(range(0,intens_mtx.shape[0],10)):
                shap_values = explainer.shap_values(intens_mtx[i:i+10,:])
                SHAP_values.append(shap_values)
            
            SHAP_values_all = []
            for i in range(len(region_data_df['region'].unique())):
                SHAP_values_all.append(np.concatenate([shap_list[i] for shap_list in SHAP_values]))
            SHAP_values_all = np.stack(SHAP_values_all)
            np.save('./processed_data/shap_all_slice_{}.npy'.format(slice_no),SHAP_values_all)

            shap_for_umap = np.abs(SHAP_values_all).mean(0)[:,rank_shap[::-1][:20]]
            shap_umap = umap.UMAP(n_components=3,n_neighbors=30,
                min_dist=0.4,metric='cosine',random_state=19).fit_transform(shap_for_umap)
            
            #np.save('./processed_data/shap_umap_{}.pkl'.format(slice_no),shap_umap_)

            shap_umap_imgs = []
            for i in range(3):
                img = IonImg_transform(shap_umap[:,i],spec_idx_transformed[:,:,slice_no], False).T
                img = (img-shap_umap[:,i].min())/(shap_umap[:,i].max()-shap_umap[:,i].min())
                shap_umap_imgs.append(img)

            shap_umap_imgs = np.stack(shap_umap_imgs,2)

            plt.imshow(shap_umap_imgs)
            plt.savefig('./figures/shap_umap_img_slice_{}.pdf'.format(slice_no),dpi=300)
            plt.close()
            


if __name__ == '__main__':
    main(False,True)