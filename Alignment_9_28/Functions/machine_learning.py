from cellpose import models
from skimage.color import label2rgb
import cv2
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
import celltypist
import scanpy as sc
import anndata as ad
from IPython.display import display, clear_output
import os
import mplcursors
# For CNN-Mask prediciton
Image_shape=[1,5000,5000] # The model is trained on 100 spatial images with dimension of 5000 by 5000 and tested on 30 images
class ThresholdPredictorCNN(nn.Module):
            def __init__(self):
                super(ThresholdPredictorCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=4, stride=4) ## Just to speed up, harsh pooling 
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3= nn.Conv2d(64,128,kernel_size=3,padding=1)
                self.fc1 = nn.Linear(128 * (Image_shape[1] // 64) * (Image_shape[2] // 64), 64)
                self.fc2 = nn.Linear(64, 1)


            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(-1, 128 * (Image_shape[1] // 64) * (Image_shape[2] // 64))
                x = torch.relu(self.fc1(x))
                x =  torch.div(0.5, (1+torch.exp(-self.fc2(x))))
                return x

        # because the model of linear layer is too large, the model is split into two and combined later
def combine_state_dict(file_path, num_chunks):
            try:
                combined_buffer = io.BytesIO()
                
                # Read each chunk file and write it to the combined buffer
                for i in range(1, num_chunks + 1):
                    chunk_file_path = f'{file_path}.part{i}'
                    with open(chunk_file_path, 'rb') as f:
                        chunk = f.read()
                        combined_buffer.write(chunk)
                        print(f"Read chunk {i} from {chunk_file_path}, size: {len(chunk)} bytes")
                
                combined_buffer.seek(0)

                # Deserialize the combined buffer back into a state dictionary
                state_dict = torch.load(combined_buffer)
                print("Model successfully reassembled from parts.")

                return state_dict

            except FileNotFoundError:
                print(f"One of the files not found: {file_path}")
            except Exception as e:
                print(f"An error occurred: {e}")
class MachineLearning:
    def run_CNN(self):
        clear_output()
        display(self.dropdown)
        plt.show()
        # Call the function with the file path
        print('resizing image to fit CNN-model')
        
        image = np.array(
            Image.fromarray(self.npy_high_thumbnail).resize(
                (int(Image_shape[2]),
                int(Image_shape[1])),
                resample=Image.LANCZOS
            )
        )
        kernel = np.outer(signal.windows.gaussian(100, 25),
                  signal.windows.gaussian(100, 25))
        convolv_matrix = signal.fftconvolve(image, kernel, mode='same')
        convolv_matrix = convolv_matrix / convolv_matrix.max()
        convolv_matrix[convolv_matrix<0]=0 # assign 0 to value smaller than 0
        image=convolv_matrix
        model=ThresholdPredictorCNN()
        print('Load Model')
        state_dict = combine_state_dict('Functions/CNN_IOU_Model.pth', 2)
        model.load_state_dict(state_dict)
        model.eval()
        # Convert numpy array to torch tensor
        image = torch.from_numpy(image).float()

        # Move the tensor to the appropriate device
        device = torch.device( "cpu")
        image = image.to(device)

        # Ensure your model is also on the same device
        model = model.to(device)
        print('Predicting mask threshold')
        # Process each image independently
        with torch.no_grad():  # Disable gradient calculation for inference
            single_image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            output = np.array(model(single_image))
        print('Predicted output:'+str(output))
        mask=convolv_matrix>output

        print('Saving mask as mask.png')
        binary_mask_255 = (mask * 255).astype(np.uint8)  # Scale 0 and 1 to 0 and 255

        # Convert to a PIL image
        image = Image.fromarray(binary_mask_255)
        # Save the image as a PNG
        if not os.path.exists('output'):
            os.makedirs('output')
            print(f"Folder output created.")
        image.save('output/mask.png')
        self.ax_tif.imshow(mask)
        self.ax_tif.set_title('CNN_predicted Mask')
        self.ax_npy.imshow(convolv_matrix)
        self.ax_npy.set_title('Blurred Spatial Image')
    def run_cellpose(self):
        model = models.Cellpose(model_type='cyto', gpu=self.GPU_MODE)
        print('Use GPU') if self.GPU_MODE else print('Use CPU')
        print('Running on HnE')
        tif_masks, flows, styles, diams = model.eval(self.tif_tile, diameter=None, channels=[0,0])
        tif_grayscale = (self.tif_tile - np.min(self.tif_tile)) / (np.max(self.tif_tile) - np.min(self.tif_tile))
        tif_overlay = label2rgb(tif_masks, image=tif_grayscale, bg_label=0, alpha=0.3, image_alpha=1.0, kind='overlay')
        self.ax_tif.clear()
        self.ax_tif.set_title("HnE Image")
        self.ax_tif.imshow(tif_overlay, extent=self.tif_origin)
        factor = self.tif_scale_factor[1] if self.tif_zoom_mode else self.tif_scale_factor[0]
        self.update_scatter(self.tif_points_dict, self.ax_tif, factor)
        self.draw_relative_grid(self.ax_tif, x_divs=5, y_divs=5)
        print('Running on Spatial')
        npy_new=self.npy_tile
        
        result_mask = cv2.resize(tif_masks, (npy_new.shape[1],npy_new.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(result_mask.shape)
        npy_overlay = label2rgb(result_mask, image=npy_new, bg_label=0, alpha=0.3, image_alpha=1.0, kind='overlay')
        self.ax_npy.clear()
        self.ax_npy.set_title("Spatial Image")
        self.ax_npy.imshow(npy_overlay,extent=self.npy_origin)
        factor= self.npy_scale_factor[1] if self.npy_zoom_mode else self.npy_scale_factor[0]
        self.update_scatter(self.npy_points_dict,self.ax_npy,factor)
        self.draw_relative_grid(self.ax_npy, x_divs=5, y_divs=5)  # 5x5 grid
    def cell_typist_brain_rawcounts(self):
        clear_output()
        display(self.dropdown)
        plt.show()
        self.adata.var_names_make_unique()

        # Both SingleR and Celltypist follow capitalize_first_letter for all gene names
        def capitalize_first_letter(arr):
            return np.array([s.capitalize() for s in arr], dtype=object)

        self.adata.var_names=capitalize_first_letter(np.array(self.adata.var_names))
        print('Load Celltyist Model')
        model = celltypist.models.Model.load(model='Mouse_Whole_Brain.pkl')  # Specify your desired model
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        predictions = celltypist.annotate(self.adata, model=model,mode='best match')
        print('Drawing')
        self.adata.obs['cell_type_subclass'] = predictions.predicted_labels
        Cell_list=self.adata.obs['cell_type_subclass']
        #Cell_amount={np.unique(Cell_list):[sum(Cell_list==target) for target in np.unique(Cell_list)]}
        Cell_amount = {cell: sum(Cell_list == cell) for cell in np.unique(Cell_list)}
        sorted_items = sorted(Cell_amount.items(), key=lambda item: item[1],reverse=True)
        sorted_names = [name for name, _ in sorted_items]
        size=0.5
        pixel_amount_threshold=2000
        self.ax_npy.clear()
        
        for i, target in enumerate(sorted_names):
            Specific_Cell = self.adata[self.adata.obs['cell_type_subclass']==target,:]
            # Find the indices where the values are True
            if (Specific_Cell.shape[0]>pixel_amount_threshold):
                x_coords, y_coords= Specific_Cell.obs.array_col_bin,Specific_Cell.obs.array_row_bin # change to array_row if needed
                scatter =self.ax_npy.scatter(x_coords*self.npy_scale_factor[1], y_coords*self.npy_scale_factor[1], s=size,label=target,zorder=i)
        self.celltypist_mode=True
        self.ax_npy.set_title('Scatter Plot of Celltypist')
        self.ax_npy.invert_yaxis()
        self.fig_text.remove()
        self.fig_text = self.fig.text(0.5, 0.01, '1) Right click to check Cell Type 2) Middle click will return to previous view', ha='center', fontsize=12)

# Adding title and labels
    
        # immature OB neurons  FOR OB-IMN GABA
        # intratelencephalic extratelencephalic it-et
        # lateral septal complex



        







