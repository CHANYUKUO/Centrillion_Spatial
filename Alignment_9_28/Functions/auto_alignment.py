from scipy import signal
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
from tqdm.notebook import tqdm
from skimage.metrics import structural_similarity as ssim

from multiprocessing import Pool
from functools import partial


class AutoAlignment: 
    @staticmethod
    def ssim_wrapper(moved_img, shifted_img):
        return ssim(shifted_img, moved_img, data_range=shifted_img.max() - shifted_img.min())

    def run_auto(self, tif_tile, npy_tile, npy_origin, tif_origin, step=50,step_size=1,whole_mode=False):
        mtx = npy_tile
        kernel = np.outer(signal.windows.gaussian(5, 2), signal.windows.gaussian(5, 2))
        convolv_matrix = signal.fftconvolve(mtx, kernel, mode='same')
        convolv_matrix = convolv_matrix / convolv_matrix.max()
        convolv_matrix[convolv_matrix < 0] = 0
        npy_tile_copy = convolv_matrix
        min_value = np.percentile(tif_tile, 1)
        max_value = 224
        tif_tile_copy = (tif_tile - min_value) / (max_value - min_value) * 255
        tif_tile_copy = tif_tile_copy.astype(np.uint8)
        inverted_image = cv2.bitwise_not(tif_tile_copy)
        alpha = 2.0
        beta = -100
        enhanced_image = cv2.convertScaleAbs(inverted_image, alpha=alpha, beta=beta)
        enhanced_image[tif_tile > 230] = 0
        cut_off=50 if self.after_transformed else 0
        if np.mean(enhanced_image) > cut_off:
            tif_tile_copy = np.array(
                Image.fromarray(enhanced_image).resize(
                    (int(npy_tile.shape[1]),
                    int(npy_tile.shape[0])),
                    resample=Image.LANCZOS
                )
            )
            mtx = tif_tile_copy
            kernel = np.outer(signal.windows.gaussian(5, 2), signal.windows.gaussian(5, 2))
            convolv_matrix = signal.fftconvolve(mtx, kernel, mode='same')
            convolv_matrix = convolv_matrix / convolv_matrix.max()
            convolv_matrix[convolv_matrix < 0] = 0
            tif_tile_copy = convolv_matrix
            matched = exposure.match_histograms(tif_tile_copy, npy_tile_copy)
            tif_tile_copy = matched
            original_img_uint8 = (npy_tile_copy * 255).astype(np.uint8)
            moved_img_uint8 = (tif_tile_copy * 255).astype(np.uint8)
            best_ssim = -1
            best_shift = (0, 0)
            shift_range = step
            y_shifts = np.arange(-shift_range, shift_range + 1, step_size)
            x_shifts = np.arange(-shift_range, shift_range + 1, step_size)
            y_shifts, x_shifts = np.meshgrid(y_shifts, x_shifts)
            print('Shift all matrices')
            shifts = np.stack([np.column_stack((np.ones(x_shifts.size), np.zeros(x_shifts.size), x_shifts.ravel())),
                            np.column_stack((np.zeros(y_shifts.size), np.ones(y_shifts.size), y_shifts.ravel()))], axis=1)
            shifted_images = np.array([cv2.warpAffine(original_img_uint8, M, (original_img_uint8.shape[1], original_img_uint8.shape[0])) 
                                    for M in shifts])
            print('Evaluate SSIM')
            with Pool() as pool:
                ssim_partial = partial(self.ssim_wrapper, moved_img_uint8)
                ssim_values = np.array(pool.map(ssim_partial, shifted_images))
            
            # Find the best shift
            best_ssim_index = np.argmax(ssim_values)
            best_shift = (x_shifts.ravel()[best_ssim_index], y_shifts.ravel()[best_ssim_index])
            point_number = len(self.npy_points_dict) + 1
            factor = self.npy_scale_factor[1] if self.npy_zoom_mode else self.npy_scale_factor[0]
            x_min, x_max = npy_origin[0], npy_origin[1]
            y_min, y_max = npy_origin[2], npy_origin[3]
            mid_x = (x_min + x_max) / 2
            mid_y = (y_min + y_max) / 2
            self.npy_points_dict[point_number] = (mid_x/factor-best_shift[0]/factor, mid_y/factor-best_shift[1]/factor)
            point_number = len(self.tif_points_dict) + 1
            factor = self.tif_scale_factor[1] if self.tif_zoom_mode else self.tif_scale_factor[0]
            x_min, x_max = tif_origin[0], tif_origin[1]
            y_min, y_max = tif_origin[2], tif_origin[3]
            mid_x = (x_min + x_max) / 2
            mid_y = (y_min + y_max) / 2
            self.tif_points_dict[point_number] = (mid_x/factor, mid_y/factor)
            self.ax_history.append('Vector')
            self.run_vector(whole_mode=whole_mode)

    def run_auto_whole(self):
        self.zoom_out()
        self.npy_zoom_mode = True
        self.tif_zoom_mode = True
        factor = self.npy_scale_factor[0] / self.npy_scale_factor[1]
        (row, cols,step,step_size) = (20, 20,100,1) if self.after_transformed else (4,4,1000,50) 
        tif_thumbnail_tiles, tif_origin_list = self.split_image_into_grid(self.tif_high_thumbnail,row,cols)
        npy_thumbnail_tiles, npy_origin_list = self.split_image_into_grid(self.npy_high_thumbnail,row,cols)
        for i in range(len(tif_thumbnail_tiles)):
            self.run_auto(tif_thumbnail_tiles[i], npy_thumbnail_tiles[i], npy_origin_list[i], tif_origin_list[i], step,step_size=step_size,whole_mode=True)
            self.show_rectangle(npy_origin_list[i][0]*factor, npy_origin_list[i][2]*factor,
                                npy_origin_list[i][1]*factor-npy_origin_list[i][0]*factor,
                                npy_origin_list[i][3]*factor-npy_origin_list[i][2]*factor)
            self.fig.canvas.draw()
        self.npy_zoom_mode = False
        self.tif_zoom_mode = False
        print('Run TPS')
        self.run_tps() if self.after_transformed else self.run_affine()

    def split_image_into_grid(self, image, rows=20, cols=20):
        if isinstance(image, np.ndarray):
            img_height, img_width = image.shape[:2]
        elif isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            raise TypeError("Unsupported image type")
        tile_width = img_width // cols
        tile_height = img_height // rows
        tiles = []
        origin_list = []
        for row in range(rows):
            for col in range(cols):
                left = col * tile_width
                upper = row * tile_height
                right = (col + 1) * tile_width
                lower = (row + 1) * tile_height
                if isinstance(image, np.ndarray):
                    tile = image[upper:lower, left:right]
                elif isinstance(image, Image.Image):
                    tile = image.crop((left, upper, right, lower))
                origin_list.append([left, right, lower, upper])
                tiles.append(tile)
        return tiles, origin_list