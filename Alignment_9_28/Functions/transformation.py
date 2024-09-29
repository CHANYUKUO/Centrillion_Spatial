from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import estimate_transform, warp, ThinPlateSplineTransform
from PIL import Image
from tqdm.notebook import tqdm
class Transformation:

    def run_vector(self,whole_mode=False):
        clear_output()
        display(self.dropdown)
        plt.show()
        tif_points = np.array([self.tif_points_dict[key] for key in sorted(self.tif_points_dict.keys())])
        npy_points = np.array([self.npy_points_dict[key] for key in sorted(self.npy_points_dict.keys())])
        self.vector_points.append([np.mean(tif_points[:,0]), np.mean(tif_points[:,1]), np.mean(npy_points[:,0]), np.mean(npy_points[:,1])])
        print([np.mean(tif_points[:,0]), np.mean(tif_points[:,1]), np.mean(npy_points[:,0]), np.mean(npy_points[:,1])])
        self.tif_points_dict = {}
        self.npy_points_dict = {}
        factor = self.npy_scale_factor[1] if self.npy_zoom_mode else self.npy_scale_factor[0]
        factor = self.npy_scale_factor[0] if whole_mode else self.npy_scale_factor[1]
        for vector in self.vector_points:
            point_B = (vector[0]/self.x_factor*factor, vector[1]/self.x_factor*factor)
            point_A = (vector[2]*factor, vector[3]*factor)
            self.ax_npy.annotate('', xy=point_B, xytext=point_A, arrowprops=dict(facecolor='green', shrink=0.05,
                                                                            width=2,
                                                                            headwidth=5,
                                                                            headlength=15
                                                                            ))
        self.ax_history.append('Vector')
        print(self.vector_points)

    

    def run_affine(self):
        clear_output()
        display(self.dropdown)
        plt.show()
        if len(self.vector_points) == 0:
            tif_points = np.array([self.tif_points_dict[key] for key in sorted(self.tif_points_dict.keys())])
            npy_points = np.array([self.npy_points_dict[key] for key in sorted(self.npy_points_dict.keys())])
            self.vector_points = [[tif_points[i,0], tif_points[i,1], npy_points[i,0], npy_points[i,1]] for i in range(tif_points.shape[0])]
        self.vector_points = np.array(self.vector_points)
        tif_points = self.vector_points[:,0:2] * self.tif_scale_factor[1]
        npy_points = self.vector_points[:,2:4] * self.tif_scale_factor[1] * self.x_factor
        print('Estimating....')
        tform = estimate_transform('affine', tif_points, npy_points)
        print('Warping....')
        max_value = np.max(self.tif_high_thumbnail)
        self.tif_high_thumbnail = warp(self.tif_high_thumbnail, tform.inverse, output_shape=tuple(np.array(self.npy_high_thumbnail.shape) / self.npy_scale_factor[1] * self.x_factor * self.tif_scale_factor[1]), cval=max_value)

        print('Calculating Low Resolution....')
        
        self.tif_thumbnail = np.array(
            Image.fromarray(self.tif_high_thumbnail).resize(
                (int(self.tif_high_thumbnail.shape[1]/self.tif_scale_factor[1]*self.tif_scale_factor[0]),
                int(self.tif_high_thumbnail.shape[0]/self.tif_scale_factor[1]*self.tif_scale_factor[0])),
                resample=Image.LANCZOS
            )
        )

        print('Done!')
        self.vector_points = []
        self.tif_points_dict = {}
        self.npy_points_dict = {}
        vmin = np.percentile(self.tif_thumbnail, 1)
        vmax = np.percentile(self.tif_thumbnail, 98)
        self.ax_tif.clear()
        self.ax_tif.set_title("HnE Image")
        self.ax_tif.imshow(self.tif_thumbnail, vmin=vmin, vmax=vmax, cmap='gray')
        self.tif_tile = self.tif_thumbnail
        self.tif_origin=[self.ax_tif.get_xlim()[0], self.ax_tif.get_xlim()[1],self.ax_tif.get_ylim()[0], self.ax_tif.get_ylim()[1]]

    def run_tps(self):
        clear_output()
        display(self.dropdown)
        plt.show()
        if len(self.vector_points) == 0:
            tif_points = np.array([self.tif_points_dict[key] for key in sorted(self.tif_points_dict.keys())])
            npy_points = np.array([self.npy_points_dict[key] for key in sorted(self.npy_points_dict.keys())])
            self.vector_points = [[tif_points[i,0], tif_points[i,1], npy_points[i,0], npy_points[i,1]] for i in range(tif_points.shape[0])]
        self.vector_points = np.array(self.vector_points)
        tif_points = self.vector_points[:,0:2] * self.tif_scale_factor[1]
        npy_points = self.vector_points[:,2:4] * self.tif_scale_factor[1] * self.x_factor
        print('Estimating....')
        tps = ThinPlateSplineTransform()
        tps.estimate(npy_points, tif_points)
        print('Warping....')
        max_value = np.max(self.tif_high_thumbnail)
        with tqdm(total=1, desc="Warping Image", ncols=100) as pbar:
            self.tif_high_thumbnail = warp(self.tif_high_thumbnail, tps,  output_shape=tuple(np.array(self.npy_high_thumbnail.shape) / self.npy_scale_factor[1] * self.x_factor * self.tif_scale_factor[1]), cval=max_value)
            pbar.update(1)
        print('Calculating Low Resolution....')
        with tqdm(total=1, desc="Resizing Image", ncols=100) as pbar:
            self.tif_thumbnail = np.array(
                Image.fromarray(self.tif_high_thumbnail).resize(
                    (int(self.tif_thumbnail.shape[1]),
                    int(self.tif_thumbnail.shape[0])),
                    resample=Image.LANCZOS
                )
            )
            pbar.update(1)
        print('Done!')
        self.vector_points = []
        self.tif_points_dict = {}
        self.npy_points_dict = {}
        vmin = np.percentile(self.tif_thumbnail, 1)
        vmax = np.percentile(self.tif_thumbnail, 98)
        self.ax_tif.clear()
        self.ax_tif.set_title("HnE Image")
        self.ax_tif.imshow(self.tif_thumbnail, vmin=vmin, vmax=vmax, cmap='gray')
        self.tif_origin=[self.ax_tif.get_xlim()[0], self.ax_tif.get_xlim()[1],self.ax_tif.get_ylim()[0], self.ax_tif.get_ylim()[1]]
