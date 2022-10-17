import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

from bokeh.plotting import figure, show
from bokeh.palettes import viridis, magma, inferno, cividis
from bokeh.models import ColumnDataSource

class charts1D_bokeh():
    def __init__(self, data, title='chart', x_label='x axis', y_label='y axis'):
        self.data = data
        self.title = title
        self.bkg_fill_color = "#fafafa"
        self.plot_height = 300
        self.x_label = x_label
        self.y_label = y_label

    def draw(self, is_show='False'):
        self.plot = figure(title=self.title, background_fill_color=self.bkg_fill_color,
                           plot_height=self.plot_height)
        self.plot.xaxis.axis_label = self.x_label
        self.plot.yaxis.axis_label = self.y_label

        for row in self.data:
            self.plot.line(x=range(self.data[0].shape[0]),
                            y=row)#,
                           #line_color='rgba(0, 0, 200, 0.5)')

        if is_show:
            show(self.plot)
class map2D():
    def __init__(self, rt, mz, intens):
        self.data = pd.DataFrame({'rt': list(rt), 'mz': list(mz), 'intens': list(intens)})
        self._transperancy = 0.7
        self._alphas = None
        self.color = (0, 0.5, 0)

    @property
    def transperancy(self):
        return self._transperancy

    @transperancy.setter
    def transperancy(self, t):
        self._transperancy = t
        max_i = self.data['intens'].max()**self._transperancy
        self._alphas = (self.data['intens'])**self._transperancy / max_i

    def draw_map(self):
        pass

class massSpec():
    def __init__(self, ms2, intens, title='ms spectra'):
        self.title = title
        self.ms2 = ms2
        self.intens = intens

    def draw_spec(self, is_show=True):

        plt.title(self.title)
        plt.xlabel('mz')
        plt.ylabel('intensity')

        for x, y in zip(self.ms2, self.intens):
            plt.plot([x, x], [0, y], '-', color='black')

        if is_show:
            plt.show()

class simple_ms_map(map2D):
    def __init__(self, rt, mz, intens, title='lcms data'):
        super().__init__(rt, mz, intens)

        self.transperancy = 0.6
        self.title = title

    def draw_map(self, is_show=True):

        plt.title(self.title)
        plt.scatter(self.data['rt'], self.data['mz'], s=3, color=self.color, alpha=self._alphas)
        plt.xlabel('retention time, s')
        plt.ylabel('mass / z')
        plt.grid(True)

        self.is_show = is_show

        if self.is_show:
            plt.show()

class matplotlib_lcms2Dmap(map2D):
    def __init__(self, rt, mz, intens, title='lcms data', x_labe='Retention time, s', y_label='Mass/Charge'):
        super().__init__(rt, mz, intens)

        self.transperancy = 0.6
        self.title = title
        self.x_label = x_labe
        self.y_label = y_label

    def draw_map(self, is_show=True):

        plt.title(self.title)
        plt.scatter(self.data['rt'], self.data['mz'], s=3, color=self.color, alpha=self._alphas)
        plt.xlabel(f'{self.x_label}')
        plt.ylabel(f'{self.y_label}')
        plt.grid(True)

        self.is_show = is_show

        if self.is_show:
            plt.show()

class labeled_ms_map(map2D):
    def __init__(self, rt, mz, intens, labels, title='lcms data'):
        super().__init__(rt, mz, intens)

        self.transperancy = 0.3
        self.title = title

        self.data['label'] = labels
        self.data['alpha'] = self._alphas

        self.labels = list(set(labels))
        if len(self.labels) > 0:
            self.colors = [self.get_color(i + 1) for i in range(len(self.labels))]
        else:
            self.colors = [(0, 0.5, 0)]

    def get_color(self, n):
        count = 200
        while n > 2:
            n -= 3
            count += 20

        c = [0, 0, 0]
        c[n] = count / 255
        return tuple(c)

    def draw_map(self, is_show=True):

        for i, label in enumerate(self.labels):
            rt = self.data[self.data['label'] == label]['rt']
            mz = self.data[self.data['label'] == label]['mz']
            al = self.data[self.data['label'] == label]['alpha']

            if label == 'unknown':
                plt.scatter(rt, mz, s=4, color='gray', alpha=al, label=label)

        plt.title(self.title)
        for i, label in enumerate(self.labels):
            rt = self.data[self.data['label'] == label]['rt']
            mz = self.data[self.data['label'] == label]['mz']
            al = self.data[self.data['label'] == label]['alpha']

            if label != 'unknown':
                if label == 'main':
                    plt.scatter(rt, mz, s=16, color='red', alpha=al, label=label)
                elif label.find('n - 1') > -1:
                    plt.scatter(rt, mz, s=6, color='blue', alpha=al, label=label)
                else:
                    plt.scatter(rt, mz, s=4, color=tuple(np.random.rand(3)), alpha=al, label=label)

        plt.xlabel('retention time, s')
        plt.ylabel('mass / z')
        plt.grid(True)
        plt.legend()

        self.is_show = is_show

        if self.is_show:
            plt.show()

class clusters_ms_map(map2D):
    def __init__(self, rt, mz, intens, classes, title='lcms data'):
        super().__init__(rt, mz, intens)
        self.classes = [int(c) for c in classes]

        self.transperancy = 0.6
        self.title = title

    def draw_map(self, is_show=True):

        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        plt.title(self.title)
        plt.scatter(self.data['rt'], self.data['mz'], c=self.classes, s=3, alpha=self._alphas, cmap=cmap)

        for i, s in enumerate(self.classes):
            if i % 5 == 0:
                plt.text(self.data['rt'].loc[i], self.data['mz'].loc[i], str(s))

        plt.xlabel('retention time, s')
        plt.ylabel('mass / z')
        plt.grid(True)

        self.is_show = is_show

        if self.is_show:
            plt.show()

class view_intens_map(simple_ms_map):
    def __init__(self, map, title='lcms data'):
        data = np.array([[i, j, map[i][j]] for i in range(len(map)) for j in range(len(map[i])) if map[i][j] > 0])
        super().__init__(data[:, 1], data[:, 0], data[:, 2], title=title)

        self.transperancy = 0.6
        self.title = title

    def draw_map(self, is_show=True):

        plt.title(self.title)
        plt.scatter(self.data['rt'], self.data['mz'], s=3, color=self.color, alpha=self._alphas)
        plt.xlabel('retention time, s')
        plt.ylabel('mass / z')
        plt.grid(True)

        self.is_show = is_show

        if self.is_show:
            plt.show()

class bokeh_ms_spectra(map2D):
    def __init__(self, rt, mz, intens, rt_position=200, title='lcms data', rt_round=True):
        if rt_round:
            rt = [round(i, 0) for i in rt]
        super().__init__(rt, mz, intens)

        self.rt_position = rt_position
        self.title = title
        self.bkg_fill_color = "#fafafa"

        if rt_position >= 0:
            data = self.data[self.data['rt'] == rt_position]
        else:
            data = self.data
        self.mz_data, self.int_data = data.values[:, 1], data.values[:, 2]
        self.plot_height = 200


    def draw_map(self, is_show=True):

        self.plot = figure(title=self.title, background_fill_color=self.bkg_fill_color,
                           plot_height=self.plot_height)
        self.plot.xaxis.axis_label = 'Mass / Z'
        self.plot.yaxis.axis_label = 'Intensity'

        for x, y in zip(self.mz_data, self.int_data):
            self.plot.line(x=[x, x],
                           y=[0, y],
                           color='rgba(0, 0, 200, 0.5)')

        if is_show:
            show(self.plot)

class bokeh_ms_spectra_simple(map2D):
    def __init__(self, mz, intens, title='lcms data'):
        self.title = title
        self.bkg_fill_color = "#fafafa"
        self.mz_data, self.int_data = mz, intens

        self.plot = figure(title=self.title, background_fill_color=self.bkg_fill_color,
                           plot_height=300)

        self.plot.title.align = "center"


    def draw_map(self, is_show=True):

        self.plot.title.text_font_size = '20pt'
        self.plot.xaxis.axis_label_text_font_size = "16pt"
        self.plot.yaxis.axis_label_text_font_size = "16pt"
        self.plot.xaxis.axis_label = 'Mass / charge'
        self.plot.yaxis.axis_label = 'Intensity'

        for x, y in zip(self.mz_data, self.int_data):
            self.plot.line(x=[x, x],
                           y=[0, y],
                           color='rgba(0, 0, 200, 0.5)')

        if is_show:
            show(self.plot)

class bokeh_ms_map(map2D):
    def __init__(self, rt, mz, intens, rt_position=200, title='lcms data', corner_points={'rt':[], 'mz':[]}, rt_round=True):
        if rt_round:
            rt = [round(i, 0) for i in rt]
        super().__init__(rt, mz, intens)

        self.corner_points = corner_points

        self.rt_position = rt_position
        self.transperancy = 0.4
        self.title = title
        self.bkg_fill_color = "#fafafa"

        self.xaxis_label = 'Retention time, s'
        self.yaxis_label = 'Mass / charge'

        self.colors = [f'rgba(20, 150, 0, {i})' for i in self._alphas]

        self.plot = figure(title=self.title, background_fill_color=self.bkg_fill_color)

        #self.Select = BoxSelectTool()
        #self.plot.add_tools(self.Select)


    def draw_map(self, is_show=True):

        self.plot.xaxis.axis_label = self.xaxis_label
        self.plot.yaxis.axis_label = self.yaxis_label

        self.plot.scatter(x=self.data['rt'], y=self.data['mz'], color=self.colors, line_color=None)
        if self.rt_position > 0:
            self.plot.line(x=np.array([self.rt_position, self.rt_position]),
                           y=np.array([0., self.data['mz'].max()]),
                           color='rgba(200, 0, 0, 0.5)')

        if is_show:
            show(self.plot)

class bokeh_ms_map_class(map2D):
    def __init__(self, rt, mz, intens, classes, names,
                 rt_position=200, title='lcms data'):
        super().__init__(rt, mz, intens)

        self.data['class'] = list(classes)
        self.data['name'] = list(names)

        self.rt_position = rt_position
        self.transperancy = 0.4
        self.title = title
        self.bkg_fill_color = "#fafafa"

        self.xaxis_label = 'Retention time, s'
        self.yaxis_label = 'Mass / charge'

        self.colors = [f'rgba(20, 150, 0, {i})' for i in self._alphas]

        self.plot = figure(title=self.title, background_fill_color=self.bkg_fill_color)
        self.plot.title.align = "center"

    def __getEllips_params(self):
        self.ellipse = []
        #print(list(set(self.data['class'])))
        for c in list(set(self.data['class'])):
            df = self.data[self.data['class'] == c]
            d = {}
            d['mz'] = df['mz'].max()
            d['rt'] = df['rt'].mean()
            d['width'] = df['rt'].max() - df['rt'].min()
            d['height'] = 3
            d['class_name'] = df['name'].loc[df.index[0]]

            if d['class_name'].find('/') > -1:
                d['fill_color'] = 'orange'
                d['color'] = 'orange'
                d['alpha'] = 0.5
                d['text_color'] = 'brown'
            else:
                d['fill_color'] = 'blue'
                d['color'] = 'blue'
                d['alpha'] = 0.1
                d['text_color'] = 'black'

            self.ellipse.append(d)
        self.ellipse = pd.DataFrame(self.ellipse)
        return self.ellipse


    def draw_map(self, is_show=True):

        self.plot.xaxis.axis_label = self.xaxis_label
        self.plot.yaxis.axis_label = self.yaxis_label
        self.plot.title.text_font_size = '20pt'
        self.plot.xaxis.axis_label_text_font_size = "16pt"
        self.plot.yaxis.axis_label_text_font_size = "16pt"

        self.__getEllips_params()

        self.plot.ellipse(x=self.ellipse['rt'], y=self.ellipse['mz'],
                          width=self.ellipse['width'], height=self.ellipse['height'],
                          fill_color=self.ellipse['fill_color'], angle=0, fill_alpha = self.ellipse['alpha'],
                          color=self.ellipse['color'])

        self.plot.text(x=self.ellipse['rt'] - self.ellipse['width'] / 2, y=self.ellipse['mz'] + 2,
                       text=self.ellipse['class_name'], angle=0, text_color=self.ellipse['text_color'])

        self.plot.scatter(x=self.data['rt'], y=self.data['mz'], color=self.colors, line_color=None)

        if self.rt_position > 0:
            self.plot.line(x=np.array([self.rt_position, self.rt_position]),
                           y=np.array([0., self.data['mz'].max()]),
                           color='rgba(200, 0, 0, 0.5)')

        if is_show:
            show(self.plot)

class bokeh_mass_map(bokeh_ms_map):
    def __init__(self,  rt, mz, intens, rt_position=200, title='lcms data', corner_points={'rt':[], 'mz':[]},
                 colorMap='Old'):
        super().__init__(rt, mz, intens, rt_position, title, corner_points)
        self.colorMap = colorMap
        self.__set_colors()

    def __set_colors(self):
        norm_intens = np.log(self.data['intens'].values)
        #norm_intens = np.log2(self.data['intens'].values)
        #norm_intens = np.sqrt(self.data['intens'].values)
        norm_intens = (norm_intens - np.min(norm_intens)) / (np.max(norm_intens) - np.min(norm_intens))
        norm_intens = [int(round(i*205 + 50)) for i in norm_intens]

        #viridis, magma, inferno, cividis
        if self.colorMap == 'viridis':
            all_colors = viridis(256)
        elif self.colorMap == 'magma':
            all_colors = magma(256)
        elif self.colorMap == 'inferno':
            all_colors = inferno(256)
        elif self.colorMap == 'cividis':
            all_colors = cividis(256)
        else:
            all_colors = []

        if len(all_colors) > 0:
            self.colors = [all_colors[i] for i in norm_intens]

    def draw_map(self, is_show=True):

        self.plot.xaxis.axis_label = self.xaxis_label
        self.plot.yaxis.axis_label = self.yaxis_label

        if len(self.corner_points['rt']) > 0 and len(self.corner_points['mz']) > 0:
            cp_x = np.array([self.corner_points['rt'][0], self.corner_points['rt'][0],
                            self.corner_points['rt'][1], self.corner_points['rt'][1]])

            cp_y = np.array([self.corner_points['mz'][0], self.corner_points['mz'][1],
                            self.corner_points['mz'][0], self.corner_points['mz'][1]])

            self.plot.scatter(x=cp_x, y=cp_y, color=self.bkg_fill_color, line_color=None)


        self.plot.scatter(x=self.data['rt'], y=self.data['mz'], color=self.colors, line_color=None)
        if self.rt_position > 0:
            self.plot.line(x=np.array([self.rt_position, self.rt_position]),
                           y=np.array([0., self.data['mz'].max()]),
                           color='rgba(200, 0, 0, 0.5)')

        if is_show:
            show(self.plot)



def main():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    #print(data[:, 1])
    map = simple_ms_map(data[:, 0], data[:, 1], data[:, 2])
    #map = bokeh_ms_map(data[:, 0], data[:, 1], data[:, 2])
    map.draw_map()

def test1():
    import mzdatapy as mzpy
    spec = mzpy.mzdata('/home/alex/Documents/LCMS/oligos/synt/220622/dT18_c2_4.mzdata.xml')
    data, vec = spec.mzdata2tab()
    # viridis, magma, inferno, cividis
    viewer = bokeh_mass_map(data[:, 0], data[:, 1], data[:, 2], rt_position=0, title='LCMS 2D map',
                                  corner_points={'rt': [0, 1500], 'mz': [100, 2000]}, colorMap='cividis')
    viewer.draw_map(is_show=True)


if __name__ == '__main__':
    test1()