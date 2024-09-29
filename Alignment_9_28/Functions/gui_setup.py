
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import ipywidgets as widgets
from IPython.display import display
class GUISetup:
    
    def setup_gui(self):
        self.fig, (self.ax_tif, self.ax_npy) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig_text=self.fig.text(0.5, 0.01, '1) Left Click: Zoom in 2) Right click: Markers 3) Middle Click:zoom out', ha='center', fontsize=12)
        self.setup_images()
        self.setup_buttons()
        self.setup_dropdown()
        self.connect_events()

    def setup_images(self):
        vmin = np.percentile(self.tif_thumbnail, 1)
        vmax = np.percentile(self.tif_thumbnail, 98)
        self.ax_tif.imshow(self.tif_thumbnail, vmin=vmin, vmax=vmax, cmap='gray')
        self.ax_tif.set_title("HE Image")

        vmin = np.percentile(self.npy_thumbnail, 1)
        vmax = np.percentile(self.npy_thumbnail, 98)
        self.ax_npy.imshow(self.npy_thumbnail, vmin=vmin, vmax=vmax, cmap='gray')
        self.ax_npy.set_title("Spatial Transcriptomic Image")

        self.tif_origin = [self.ax_tif.get_xlim()[0], self.ax_tif.get_xlim()[1], self.ax_tif.get_ylim()[0], self.ax_tif.get_ylim()[1]]
        self.npy_origin = [self.ax_npy.get_xlim()[0], self.ax_npy.get_xlim()[1], self.ax_npy.get_ylim()[0], self.ax_npy.get_ylim()[1]]

    def setup_buttons(self):
        self.del_button_ax = plt.axes([0.40, 0.92, 0.1, 0.08])
        self.del_button = Button(self.del_button_ax, 'Delete Last')
        self.del_button.on_clicked(self.del_last_point)

        self.grid_button_ax = plt.axes([0.50, 0.92, 0.1, 0.08])
        self.grid_button = Button(self.grid_button_ax, 'Grid: Off')
        self.grid_button.on_clicked(self.toggle_grid_button)

        self.sync_button_ax = plt.axes([0.60, 0.92, 0.1, 0.08])
        self.sync_button = Button(self.sync_button_ax, 'Sync: Off')
        self.sync_button.on_clicked(self.toggle_sync_button)
        

    def setup_dropdown(self):
        self.functions = {
            'Zoom out': self.zoom_out,
            'Vector Register': self.run_vector,
            'Run Cellpose': self.run_cellpose,
            'Transform Affine': self.run_affine,
            'Transform TPS': self.run_tps,
            'Overlap': lambda: self.run_overlap(self.tif_tile, self.npy_tile),
            '150 Auto': lambda: self.run_auto(self.tif_tile, self.npy_tile, self.npy_origin, self.tif_origin, step=150),
            '50 Auto': lambda: self.run_auto(self.tif_tile, self.npy_tile, self.npy_origin, self.tif_origin, step=50),
            'Auto Full Image': self.run_auto_whole,
            'CNN Mask':self.run_CNN,
            'Save Transformed HnE ':self.output_current,
            'Celltypist (L2 logistic Reg with Allen Brain)':self.cell_typist_brain_rawcounts
        }

        self.dropdown = widgets.Dropdown(
            options=self.functions.keys(),
            value=None,
            description='Tools:',
        )
        self.dropdown.observe(self.handle_dropdown_change, names='value')

    def connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.combined_button_press_event)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)


    def run(self):
        display(self.dropdown)
        plt.show()
    def handle_dropdown_change(self, change):
            selected_function = self.functions.get(change['new'])
            if selected_function:
                self.dropdown.value = None
                selected_function()
    
