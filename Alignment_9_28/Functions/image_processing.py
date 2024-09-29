from PIL import Image
import gzip
import numpy as np
import matplotlib.patches as patches 
import matplotlib.pyplot as plt
import cv2
import os
import anndata as ad
class ImageProcessing:
    def input_image(self,tif_image_path,npy_image_path,tif_scale_factor,npy_scale_factor,spatial_dataset):

        Image.MAX_IMAGE_PIXELS = None
        print('Loading Images')
        tif_image = Image.open(tif_image_path)
        tif_image_array = np.array(tif_image)  # Convert to NumPy array
        if spatial_dataset:
            print('Note: Adata objects typically takes longer to load')
            adata = ad.read_h5ad(npy_image_path)
            # Change array_row_bin to array_row according to data_struc
            array_row_bin = np.array(adata.obs.array_row_bin) 
            array_col_bin = np.array(adata.obs.array_col_bin)
            total=np.array(adata.X.sum(axis=1))
            reconstructed_matrix = np.zeros((1000,1000))
            for i in range(len(total)):
                reconstructed_matrix[array_row_bin[i],array_col_bin[i]]=total[i]
            npy_image_array=reconstructed_matrix
        else:
            with gzip.open(npy_image_path, 'rb') as f:
                npy_image_array = np.load(f)
            adata=None

        if npy_image_array.dtype != np.uint8:
            npy_image_array_normalized = (255 * (npy_image_array - np.min(npy_image_array)) / (np.max(npy_image_array) - np.min(npy_image_array))).astype(np.uint8)
        else:
            npy_image_array_normalized = npy_image_array
            
        x_factor=tif_image.size[0]/npy_image_array.shape[0]

        # Initialize dictionaries to store point coordinates for each image

        print('Compute Low-resolution')
        # Rescale the images for the initial low-res display
        tif_thumbnail = np.array(
            Image.fromarray(tif_image_array).resize(
                (int(tif_image_array.shape[1] * tif_scale_factor[0]),
                int(tif_image_array.shape[0] * tif_scale_factor[0])),
                resample=Image.LANCZOS
            )
        )
        print('--HnE Done')
        npy_thumbnail = np.array(
            Image.fromarray(npy_image_array_normalized).resize(
                (int(npy_image_array_normalized.shape[1] * npy_scale_factor[0]),
                int(npy_image_array_normalized.shape[0] * npy_scale_factor[0])),
                resample=Image.LANCZOS
            )
        )
        print('--Spatial Done')
        print('Compute high-resolution')

        tif_high_thumbnail = np.array(
            Image.fromarray(tif_image_array).resize(
                (int(tif_image_array.shape[1] * tif_scale_factor[1]),
                int(tif_image_array.shape[0] * tif_scale_factor[1])),
                resample=Image.LANCZOS
            )
        )
        print('--HnE Done')

        npy_high_thumbnail = np.array(
            Image.fromarray(npy_image_array_normalized).resize(
                (int(npy_image_array_normalized.shape[1] * npy_scale_factor[1]),
                int(npy_image_array_normalized.shape[0] * npy_scale_factor[1])),
                resample=Image.LANCZOS
            )
        )
        print('--Spatial Done')

        print('Compute HnE Grayscale')
        # Convert the HE image to grayscale
        #tif_thumbnail = tif_thumbnail[:, :, 0] * 0.5189 + tif_thumbnail[:, :, 1] * 0.1 + tif_thumbnail[:, :, 2] * 1
        tif_thumbnail = tif_thumbnail[:, :, 0] * 0.2989 + tif_thumbnail[:, :, 1] * 0.587 + tif_thumbnail[:, :, 2] * 0.114
        tif_high_thumbnail = tif_high_thumbnail[:, :, 0] * 0.2989 + tif_high_thumbnail[:, :, 1] * 0.587 + tif_high_thumbnail[:, :, 2] * 0.114
        #tif_high_thumbnail = tif_high_thumbnail[:, :, 0] * 0.5189 + tif_high_thumbnail[:, :, 1] * 0.1 + tif_high_thumbnail[:, :, 2] * 1
        print('Compute HnE Grayscale Done')
        return tif_thumbnail,tif_high_thumbnail,npy_thumbnail,npy_high_thumbnail,x_factor,adata

    def update_scatter(self, points_dict, ax, factor):
        for point_number, (x, y) in points_dict.items():
            x, y = x*factor, y*factor
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()
            is_x_in_lim = current_xlim[0] <= x <= current_xlim[1]
            is_y_in_lim = current_ylim[1] <= y <= current_ylim[0]
            if is_x_in_lim and is_y_in_lim:
                ax.scatter(x, y, color='red', s=2)
                ax.text(x, y, f'{point_number}', color='white', fontsize=12,
                        ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0, edgecolor='none'))

    def draw_relative_grid(self, ax, x_divs, y_divs):
        if self.toggle_grid == 0:
            ax.grid(False)
            for i in range(len(self.minor_grid)):
                self.minor_grid[i].remove()
            self.minor_grid = []
        else:
            ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], x_divs + 1))
            ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], y_divs + 1))
            ax.grid(True, color='blue', linestyle='-', linewidth=1.5)
            if self.toggle_grid == 2:
                x_limits = ax.get_xlim()
                y_limits = ax.get_ylim()
                for x in np.linspace(x_limits[0], x_limits[1], x_divs * 5 + 1):
                    line, = ax.plot([x, x], y_limits, color='g', linestyle='--', linewidth=0.5)
                    self.minor_grid.append(line)
                for y in np.linspace(y_limits[0], y_limits[1], x_divs * 5 + 1):
                    line, = ax.plot(x_limits, [y, y], color='g', linestyle='--', linewidth=0.5)
                    self.minor_grid.append(line)
    def display_vector(self):
        factor = self.npy_scale_factor[0] if not self.npy_zoom_mode else self.npy_scale_factor[1]
        for vector in self.vector_points:
            point_B = (vector[0]/self.x_factor*factor, vector[1]/self.x_factor*factor)
            point_A = (vector[2]*factor, vector[3]*factor)
            self.ax_npy.annotate('', xy=point_B, xytext=point_A, arrowprops=dict(facecolor='green', shrink=0.05,
                                                                            width=2,
                                                                            headwidth=5,
                                                                            headlength=15
                                                                            ))
        plt.draw()
    def show_rectangle(self, x, y, width, height):
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        self.ax_npy.add_patch(rect)
        self.ax_npy.add_patch(rect)
    
    def run_overlap(self, tif_tile, npy_tile):
        min_value = np.percentile(npy_tile, 1)
        max_value = np.percentile(npy_tile, 99)
        npy_tile_copy = (npy_tile - min_value) / (max_value - min_value)
        npy_tile_copy = np.clip(npy_tile_copy, 0, 1)
        min_value = np.percentile(tif_tile, 1)
        max_value = 224
        tif_tile_copy = (tif_tile - min_value) / (max_value - min_value) * 255
        tif_tile_copy = tif_tile_copy.astype(np.uint8)
        inverted_image = cv2.bitwise_not(tif_tile_copy)
        alpha = 2.0
        beta = -100
        enhanced_image = cv2.convertScaleAbs(inverted_image, alpha=alpha, beta=beta)
        enhanced_image[tif_tile > 230] = 0
        scaled_npy_image_array = npy_tile_copy * 255
        scaled_he_gray_down = np.array(
            Image.fromarray(enhanced_image).resize(
                (int(npy_tile_copy.shape[1]),
                int(npy_tile_copy.shape[0])),
                resample=Image.LANCZOS
            )
        )
        overlay_image = np.stack((
            scaled_he_gray_down,
            scaled_npy_image_array,
            np.zeros_like(scaled_he_gray_down)
        ), axis=-1)
        overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
        self.ax_npy.clear()
        
        self.ax_npy.imshow(overlay_image, extent=self.npy_origin)
        self.ax_npy.set_title("Spatial Image")
        self.ax_tif.imshow(enhanced_image, extent=self.tif_origin, cmap='Reds')
        self.draw_relative_grid(self.ax_tif, x_divs=5, y_divs=5)
        self.draw_relative_grid(self.ax_npy, x_divs=5, y_divs=5)
        self.display_vector()
    def output_current(self):
        if not os.path.exists('output'):
            os.makedirs('output')
            print(f"Folder output created.")
        image = Image.fromarray(self.tif_high_thumbnail)
        image = image.convert("L")
        image.save('output/HnE_transformed.png')
        print('HnE saved as HnE_transformed.png ')
    

        # Save the image as a PNG
        