import matplotlib.pyplot as plt
import numpy as np
class EventHandlers:
    def combined_button_press_event(self, event):
        if event.inaxes == self.ax_tif:
            self.onclick(event, self.ax_tif, self.tif_scale_factor, self.tif_points_dict, self.tif_zoom_mode)
        elif event.inaxes == self.ax_npy:
            self.onclick(event, self.ax_npy, self.npy_scale_factor, self.npy_points_dict, self.npy_zoom_mode)
        self.on_select(event)

    def onclick(self, event, ax, scale_factor, points_dict, zoom_mode):

        if (ax == self.ax_tif) or (ax == self.ax_npy):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None and event.button == 3:
                if self.celltypist_mode:
                    
                    factor = scale_factor[0] if ax == self.ax_tif else 1
                    x, y = int(event.xdata / factor), int(event.ydata / factor)
                    matching_cells = self.adata.obs[(self.adata.obs['array_col_bin'] == x) & 
                               (self.adata.obs['array_row_bin'] == y)]
                    if not matching_cells.empty:
                        # If a match is found, return the cell type from the 'cell_type_subclass' column
                        if self.previous_scatter_npy is not None:
                            self.previous_scatter_npy.remove()
                        if self.previous_scatter_tif is not None:
                            self.previous_scatter_tif.remove()
                        self.ax_npy.set_title(matching_cells['cell_type_subclass'].values[0])
                        self.previous_scatter_npy = self.ax_npy.scatter(x, y, color='black', s=2)
                        self.previous_scatter_tif = self.ax_tif.scatter(x*self.tif_scale_factor[0], y*self.tif_scale_factor[0], color='red', s=2)
                    
                    else:
                        print('None')  # No match found
                else:
                    self.ax_history.append(ax)
                    point_number = len(points_dict) + 1
                    factor = scale_factor[1] if zoom_mode else scale_factor[0]
                    x, y = int(event.xdata / factor), int(event.ydata / factor)
                    points_dict[point_number] = (x, y)
                    self.update_scatter(points_dict, ax, factor)
                    print(f'Point {point_number} selected on {"TIF" if ax == self.ax_tif else "NPY"}: ({x}, {y})')

    def on_select(self, event):
        zoom_mode = self.tif_zoom_mode if event.inaxes == self.ax_tif else self.npy_zoom_mode
        if zoom_mode and event.button == 1:
            self.x0, self.y0 = event.xdata, event.ydata
        if not zoom_mode and event.button == 1:
            scale = self.tif_scale_factor if event.inaxes == self.ax_tif else self.npy_scale_factor
            self.x0, self.y0 = event.xdata/scale[0]*scale[1], event.ydata/scale[0]*scale[1]
        if event.button == 2:
            self.zoom_out()

    def on_release(self, event):
        zoom_mode = self.tif_zoom_mode if event.inaxes == self.ax_tif else self.npy_zoom_mode
        button_list = [self.del_button_ax,self.grid_button_ax,self.sync_button_ax]
        if event.inaxes not in button_list and event.button == 1:
            scale = self.tif_scale_factor if event.inaxes == self.ax_tif else self.npy_scale_factor
            image = self.tif_high_thumbnail if event.inaxes == self.ax_tif else self.npy_high_thumbnail
            if not zoom_mode:
                x1, y1 = event.xdata/scale[0]*scale[1], event.ydata/scale[0]*scale[1]
            elif zoom_mode:
                x1, y1 = event.xdata, event.ydata
            event.inaxes.clear()
            zoom_lim = np.int32(np.array([min(self.x0, x1), max(self.x0, x1), max(self.y0, y1), min(self.y0, y1)]))
            tile = image[zoom_lim[3]:zoom_lim[2], zoom_lim[0]:zoom_lim[1]]
            origin_list = [min(self.x0, x1), max(self.x0, x1), max(self.y0, y1), min(self.y0, y1)]
            if event.inaxes == self.ax_tif:
                self.tif_tile = tile
                self.tif_origin = origin_list
                self.tif_zoom_mode = True
            elif event.inaxes == self.ax_npy:
                self.npy_tile = tile
                self.npy_origin = origin_list
                self.npy_zoom_mode = True
            vmin = np.percentile(tile, 1)
            vmax = np.percentile(tile, 98)
            event.inaxes.imshow(tile, extent=origin_list, vmin=vmin, vmax=vmax, cmap='gray')
            event.inaxes.figure.canvas.draw()
            if self.sync_mode:
                if event.inaxes==self.ax_tif:
                    
                    self.ax_npy.clear() 
                    self.npy_tile=self.npy_high_thumbnail[np.int32(zoom_lim[3]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1]):np.int32(zoom_lim[2]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1]), 
                                    np.int32(zoom_lim[0]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1]):np.int32(zoom_lim[1]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1])]
                    self.npy_origin=[ np.int32(zoom_lim[0]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1]),np.int32(zoom_lim[1]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1]),np.int32(zoom_lim[2]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1]),np.int32(zoom_lim[3]/self.x_factor/self.tif_scale_factor[1]*self.npy_scale_factor[1])]
                    vmin = np.percentile(self.npy_tile, 1)
                    vmax = np.percentile(self.npy_tile, 98)
                    self.ax_npy.imshow(self.npy_tile,extent=self.npy_origin, vmin=vmin, vmax=vmax, cmap='gray')
                    self.npy_zoom_mode=True
                    
                else:
                    self.ax_tif.clear()
                    self.tif_tile=self.tif_high_thumbnail[np.int32(zoom_lim[3]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1]):np.int32(zoom_lim[2]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1]), 
                                    np.int32(zoom_lim[0]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1]):np.int32(zoom_lim[1]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1])]
                    self.tif_origin=[np.int32(zoom_lim[0]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1]),np.int32(zoom_lim[1]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1]),np.int32(zoom_lim[2]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1]),np.int32(zoom_lim[3]*self.x_factor*self.tif_scale_factor[1]/self.npy_scale_factor[1])]
                    vmin = np.percentile(self.tif_tile, 1)
                    vmax = np.percentile(self.tif_tile, 98)
                    self.ax_tif.imshow(self.tif_tile,extent=self.tif_origin, vmin=vmin, vmax=vmax, cmap='gray')
                    self.tif_zoom_mode=True 
            self.draw_relative_grid(self.ax_tif, x_divs=5, y_divs=5)
            self.draw_relative_grid(self.ax_npy, x_divs=5, y_divs=5)
            self.update_scatter(self.tif_points_dict, self.ax_tif, self.tif_scale_factor[1])
            self.update_scatter(self.npy_points_dict, self.ax_npy, self.npy_scale_factor[1])
            self.display_vector()
    def del_last_point(self, event):
        if self.ax_history[-1] == self.ax_tif:
            print('del HE marker')
            last_key, last_value = self.tif_points_dict.popitem()
            factor = self.tif_scale_factor[1] if self.tif_zoom_mode else self.tif_scale_factor[0]
            self.ax_tif.clear()
            vmin = np.percentile(self.tif_tile, 1)
            vmax = np.percentile(self.tif_tile, 98)
            self.ax_tif.imshow(self.tif_tile, vmin=vmin, vmax=vmax, cmap='gray', extent=self.tif_origin)
            self.update_scatter(self.tif_points_dict, self.ax_tif, factor)
            self.draw_relative_grid(self.ax_tif, x_divs=5, y_divs=5)
        elif self.ax_history[-1] == self.ax_npy:
            print('del Spatial marker')
            last_key, last_value = self.npy_points_dict.popitem()
            factor = self.npy_scale_factor[1] if self.npy_zoom_mode else self.npy_scale_factor[0]
            self.ax_npy.clear()
            vmin = np.percentile(self.npy_tile, 1)
            vmax = np.percentile(self.npy_tile, 98)
            self.ax_npy.imshow(self.npy_tile, vmin=vmin, vmax=vmax, cmap='gray', extent=self.npy_origin)
            self.update_scatter(self.npy_points_dict, self.ax_npy, factor)
            self.draw_relative_grid(self.ax_npy, x_divs=5, y_divs=5)
        if self.ax_history[-1]=='Vector':
            if len(self.vector_points)==0:
                print('Vector is empty')
            else:
                print('Del Last Vector')
                self.vector_points.pop()
                self.display_vector()
        self.ax_history.pop()
        plt.draw()

    def toggle_grid_button(self, event):
        self.toggle_grid = self.toggle_grid + 1 if self.toggle_grid < 2 else 0
        if self.toggle_grid == 0:
            self.grid_button.label.set_text("Grid: Off")
        else:
            self.grid_button.label.set_text("Grid: On")
        self.draw_relative_grid(self.ax_tif, x_divs=5, y_divs=5)
        self.draw_relative_grid(self.ax_npy, x_divs=5, y_divs=5)
    def toggle_sync_button(self, event):
        self.sync_mode = not self.sync_mode
        if self.sync_mode == 0:
            self.sync_button.label.set_text("Sync: Off")
        else:
            self.sync_button.label.set_text("Sync: On")
    def zoom_out(self):
        self.tif_zoom_mode = False
        self.npy_zoom_mode = False
        self.ax_tif.clear()
        vmin = np.percentile(self.tif_thumbnail, 1)
        vmax = np.percentile(self.tif_thumbnail, 98)
        self.ax_tif.imshow(self.tif_thumbnail, vmin=vmin, vmax=vmax, cmap='gray')
        factor = self.tif_scale_factor[1] if self.tif_zoom_mode else self.tif_scale_factor[0]
        self.update_scatter(self.tif_points_dict, self.ax_tif, factor)
        self.ax_npy.clear()
        vmin = np.percentile(self.npy_thumbnail, 1)
        vmax = np.percentile(self.npy_thumbnail, 98)
        self.ax_npy.imshow(self.npy_thumbnail, vmin=vmin, vmax=vmax, cmap='gray')
        factor = self.npy_scale_factor[1] if self.npy_zoom_mode else self.npy_scale_factor[0]
        self.update_scatter(self.npy_points_dict, self.ax_npy, factor)
        self.draw_relative_grid(self.ax_tif, x_divs=5, y_divs=5)
        self.draw_relative_grid(self.ax_npy, x_divs=5, y_divs=5)
        self.tif_tile = self.tif_thumbnail
        self.npy_tile = self.npy_thumbnail
        self.display_vector()
        self.tif_origin=[self.ax_tif.get_xlim()[0], self.ax_tif.get_xlim()[1],self.ax_tif.get_ylim()[0], self.ax_tif.get_ylim()[1]]
        self.npy_origin=[self.ax_npy.get_xlim()[0], self.ax_npy.get_xlim()[1],self.ax_npy.get_ylim()[0], self.ax_npy.get_ylim()[1]]
        self.fig_text.remove()
        self.fig_text =self.fig.text(0.5, 0.01, '1) Left Click: Zoom in 2) Right click: Markers 3) Middle Click:zoom out', ha='center', fontsize=12)
        self.celltypist_mode=False
        plt.draw()