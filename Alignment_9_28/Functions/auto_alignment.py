from scipy import signal
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
from tqdm.notebook import tqdm
from skimage.metrics import structural_similarity as ssim
class AutoAlignment: 
    def run_auto(self, tif_tile, npy_tile, npy_origin, tif_origin, step=50,whole_mode=False):
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
        if np.mean(enhanced_image) > 50:
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
            for x_shift in tqdm(range(-shift_range, shift_range + 1), desc="X Shifts"):
                for y_shift in range(-shift_range, shift_range + 1):
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    shifted_image = cv2.warpAffine(original_img_uint8, M, (original_img_uint8.shape[1], original_img_uint8.shape[0]))
                    current_ssim = ssim(shifted_image, moved_img_uint8, data_range=shifted_image.max() - shifted_image.min())
                    if current_ssim > best_ssim:
                        best_ssim = current_ssim
                        best_shift = (x_shift, y_shift)
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
            print([mid_x, mid_y])
            print(mid_x/factor)
            print(best_ssim)

    def run_auto_whole(self):
        self.zoom_out()
        self.npy_zoom_mode = True
        self.tif_zoom_mode = True
        factor = self.npy_scale_factor[0] / self.npy_scale_factor[1]
        tif_thumbnail_tiles, tif_origin_list = self.split_image_into_grid(self.tif_high_thumbnail)
        npy_thumbnail_tiles, npy_origin_list = self.split_image_into_grid(self.npy_high_thumbnail)
        for i in tqdm(range(len(tif_thumbnail_tiles))):
            self.run_auto(tif_thumbnail_tiles[i], npy_thumbnail_tiles[i], npy_origin_list[i], tif_origin_list[i], 50,whole_mode=True)
            self.show_rectangle(npy_origin_list[i][0]*factor, npy_origin_list[i][2]*factor,
                                npy_origin_list[i][1]*factor-npy_origin_list[i][0]*factor,
                                npy_origin_list[i][3]*factor-npy_origin_list[i][2]*factor)
            self.fig.canvas.draw()
        self.npy_zoom_mode = False
        self.tif_zoom_mode = False
        print('Run TPS')
        self.run_tps()

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