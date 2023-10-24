import pandas as pd
import msvis
import oligoMSMSinterp as interp
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class data_sorber():
    def __init__(self, data):
        self.data = data
        self.rect = []
        self.masses = []
        self.text = []
        self.text_coord = []
        self.viewer = None
        self.seq_data = []
        self.interpreter = interp.deltaOligo_interpreter()

        self._alphas = None
        self.transperancy_ = 0.6
        self.transperancy = 0.6
        self.title = 'Mass data'
        self.x_label = 'Retention time'
        self.y_label = 'Mass, Da'
        self.color = (0, 0.5, 0)

        self.font = {'family': 'serif',
                     'color': 'darkred',
                     'weight': 'bold',
                     'size': 7,
                     }

        self.dist_font = {'family': 'serif',
                     'color': 'black',
                     'weight': 'bold',
                     'size': 7,
                     }

    @property
    def transperancy(self):
        return self._transperancy

    @transperancy.setter
    def transperancy(self, t):
        self._transperancy = t
        max_i = self.data['intens'].max() ** self._transperancy
        self._alphas = (self.data['intens']) ** self._transperancy / max_i

    def get_avg_mass(self, rect):
        df = self.data[(self.data['mass'] >= rect[1])&(self.data['mass'] <= rect[3])]
        df = df[(df['rt'] >= rect[0])&(df['rt'] <= rect[2])]
        return (df['mass'] * df['intens']).sum() / df['intens'].sum()

    def get_rt_interval(self, rect):
        df = self.data[(self.data['mass'] >= rect[1])&(self.data['mass'] <= rect[3])]
        df = df[(df['rt'] >= rect[0])&(df['rt'] <= rect[2])]
        return df['rt'].min(), df['rt'].max()

    def get_coord(self, rect):
        mass = self.get_avg_mass(self.rect[-1])
        rt_min, rt_max = self.get_rt_interval(self.rect[-1])
        return rt_min, rt_max, mass

    def callback(self, eclick, erelease):
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        #plt.text(self.x1, self.y1, 'X', fontdict=self.font)
        #plt.show()


    def keypress(self, event):
        #print(' Key pressed.', event.key)
        if event.key in ['delete']:
            if len(self.rect) > 0:
                print('Delete last')
                self.rect.pop()
                self.masses.pop()
                self.text.pop()
                self.text_coord.pop()
                #print(self.rect)
            if len(self.seq_data) > 0:
                self.seq_data.pop()

        if event.key in ['A', 'a']:
            self.rect.append((self.x1, self.y1, self.x2, self.y2))
            coord = self.get_coord(self.rect[-1])
            self.masses.append(coord[2])
            self.text.append(str(round(self.masses[-1],1)))
            self.text_coord.append(coord)

        if event.key in ['D', 'd']:
            if len(self.rect) > 1:
                dist = abs(self.masses[len(self.masses) - 1] - self.masses[len(self.masses) - 2])
                seq = self.interpreter.find_mass(dist)
                print(dist, seq)
                coord_1 = self.text_coord[len(self.masses) - 1]
                coord_2 = self.text_coord[len(self.masses) - 2]
                self.seq_data.append({'dist': round(dist, 1), 'seq': seq, 'coord 1': coord_1, 'coord 2': coord_2})


        if event.key in ['T', 't']:
            with open('sorber.obj', 'wb') as f:
                pickle.dump(self, f)

        if event.key in ['P', 'p']:

            rt_min, rt_max = min([self.x1, self.x2]), max([self.x1, self.x2])
            mass_min, mass_max = min([self.y1, self.y2]), max([self.y1, self.y2])

            conditions = ((self.data['mass'] >= mass_min)&(self.data['mass'] <= mass_max)&
                          (self.data['rt'] >= rt_min)&(self.data['rt'] <= rt_max))

            self.data = self.data[~conditions]

        self.draw_data(is_show=True, is_cla=True)



    def draw_data(self, is_show=False, is_cla=False):

        if is_cla:
            self.xlim = self.current_ax.get_xlim()
            self.ylim = self.current_ax.get_ylim()

            plt.cla()

            self.current_ax.set_xlim(self.xlim[0], self.xlim[1])
            self.current_ax.set_ylim(self.ylim[0], self.ylim[1])

        plt.title(self.title)
        plt.scatter(self.data['rt'], self.data['mass'], s=3, color=self.color, alpha=self._alphas)

        self.xlim = self.current_ax.get_xlim()
        self.ylim = self.current_ax.get_ylim()

        for t, coord in zip(self.text, self.text_coord):
            plt.text(coord[1], coord[2], t, fontdict=self.font)

        for elem in self.seq_data:
            txt = f"{elem['seq']}; {elem['dist']}"
            t_x = elem['coord 1'][1]
            t_y = (elem['coord 1'][2] + elem['coord 2'][2]) / 2
            plt.text(t_x, t_y, txt, fontdict=self.dist_font)
            plt.plot([elem['coord 1'][0], elem['coord 1'][1]], [elem['coord 1'][2], elem['coord 1'][2]], '-',
                     color='darkgreen')
            plt.plot([elem['coord 2'][0], elem['coord 2'][1]], [elem['coord 2'][2], elem['coord 2'][2]], '-',
                     color='darkgreen')
            plt.plot([t_x, t_x], [elem['coord 1'][2], elem['coord 2'][2]], '-',
                     color='darkgreen')

        plt.xlabel(f'{self.x_label}')
        plt.ylabel(f'{self.y_label}')
        plt.grid(True)

        if is_show:
            plt.show()

    def plot_data(self):

        self.fig, self.current_ax = plt.subplots()

        self.draw_data()

        RS = RectangleSelector(self.current_ax, self.callback,
                               useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        plt.connect('key_press_event', self.keypress)

        plt.show()


class ms2Spec_dataSorber():

    def __init__(self, spectra_dataframe, precursor_mass, chrom_rt_interval):
        self.precursor_mass = precursor_mass
        self.rt_interval = chrom_rt_interval
        self.data = spectra_dataframe

        self.title = 'MS2 spectra'
        self.x_label = 'Mass, Da'
        self.y_label = 'Intensity'
        self.color = (0.1, 0.1, 0.9)
        self.dist_bord_color = (0.1, 1, 0.1)

        self.font = {'family': 'serif',
                     'color': 'darkred',
                     'weight': 'bold',
                     'size': 9,
                     }

        self.dist_font = {'family': 'serif',
                     'color': 'blue',
                     'weight': 'bold',
                     'size': 9,
                     }

        self.max_intens = self.data['f.intens'].max()

        self.masses = []
        self.text = []
        self.text_coord = []
        self.seq_data = []

        self.interpr = interp.ms2_interp(prefix='')
        #for i, m in enumerate(self.interpr.mass_list):
        #    print(m, self.interpr.seq_list[i])

    def interp_data_in_rect(self):
        mass_min, mass_max = min([self.x1, self.x2]), max([self.x1, self.x2])
        intens_min, intens_max = min([self.y1, self.y2]), max([self.y1, self.y2])

        conditions = ((self.data['f.mass'] >= mass_min)&(self.data['f.mass'] <= mass_max)&
                      (self.data['f.intens'] >= intens_min)&(self.data['f.intens'] <= intens_max))
        df = self.data[conditions]

        for mass, intens in zip(df['f.mass'], df['f.intens']):
            matches = self.interpr.interp_fragment_by_mass(mass, treshold=2)
            self.masses.append(mass)
            text = f'{round(mass, 1)}\n'
            for m in matches:
                text += f"{m['seq']} ({m['ion type']})\n"
            self.text.append(text)
            self.text_coord.append((mass, intens))

    def get_pick_param(self, rect):
        df = self.data[(self.data['f.mass'] >= rect[0])&(self.data['f.mass'] <= rect[2])]
        mass = (df['f.mass'] * df['f.intens']).sum() / df['f.intens'].sum()
        max_intens = df['f.intens'].max()
        return mass, max_intens

    def callback(self, eclick, erelease):
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata

    def keypress(self, event):
        if event.key in ['delete']:
            if len(self.masses) > 0:
                self.masses.pop()
                self.text.pop()
                self.text_coord.pop()
            if len(self.seq_data) > 0:
                self.seq_data.pop()

        if event.key in ['A', 'a']:
            pick = self.get_pick_param((self.x1, self.y1, self.x2, self.y2))
            self.masses.append(pick)
            self.text.append(str(round(pick[0], 1)))
            x, y = pick[0], pick[1]
            self.text_coord.append((x, y))

        if event.key in ['D', 'd']:
            if len(self.masses) > 1:
                m1 = self.masses[len(self.masses)-1][0]
                m2 = self.masses[len(self.masses)-2][0]
                i1 = self.masses[len(self.masses)-1][1]
                i2 = self.masses[len(self.masses)-2][1]
                dist = abs(m1 - m2)

                seq = str(self.interpr.find_mass(dist, delta=1))

                pos = 30
                if seq.find('(B)') > -1:
                    pos = 10
                elif seq.find('(W)') > -1:
                    pos = 5
                elif seq.find('(b-w)') > -1:
                    pos = 5
                elif seq.find('(*f)') > -1:
                    pos = 3
                else:
                    pos = 30

                x = min([m1, m2])
                y = self.max_intens - self.max_intens / pos

                self.seq_data.append({'dist': round(dist, 1), 'coord': (x, y),
                                      'line1': (m1, i1 + i1/10, self.max_intens),
                                      'line2': (m2, i2 + i2/10, self.max_intens),
                                      'line3': (m1, y, m2, y),
                                      'interp': seq, 'pos': pos})

        if event.key in ['G', 'g']:
            self.interp_data_in_rect()

        self.draw_data(True, True)

    def draw_data(self, is_show=False, is_cla=False):
        if is_cla:
            self.xlim = self.current_ax.get_xlim()
            self.ylim = self.current_ax.get_ylim()

            plt.cla()

            self.current_ax.set_xlim(self.xlim[0], self.xlim[1])
            self.current_ax.set_ylim(self.ylim[0], self.ylim[1])

        plt.title(self.title)

        for x, y in zip(self.data['f.mass'], self.data['f.intens']):
            plt.plot([x, x], [0, y], '-', color=self.color)

        for t, c in zip(self.text, self.text_coord):
            plt.text(c[0], c[1], t, fontdict=self.font)

        for d in self.seq_data:
            plt.text(d['coord'][0], d['coord'][1] - d['coord'][1]/30, str(d['dist']), fontdict=self.dist_font)
            plt.text(d['coord'][0], d['coord'][1], str(d['interp']), fontdict=self.dist_font)
            plt.plot([d['line1'][0], d['line1'][0]], [d['line1'][1], d['line1'][2]], '-', color=self.dist_bord_color)
            plt.plot([d['line2'][0], d['line2'][0]], [d['line2'][1], d['line2'][2]], '-', color=self.dist_bord_color)
            plt.plot([d['line3'][0], d['line3'][2]], [d['line3'][1], d['line3'][3]], '--', color=self.dist_bord_color)

        self.xlim = self.current_ax.get_xlim()
        self.ylim = self.current_ax.get_ylim()

        plt.xlabel(f'{self.x_label}')
        plt.ylabel(f'{self.y_label}')
        plt.grid(False)

        if is_show:
            plt.show()
    def plot_data(self):
        self.fig, self.current_ax = plt.subplots()

        self.draw_data()

        RS = RectangleSelector(self.current_ax, self.callback,
                               useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        plt.connect('key_press_event', self.keypress)

        plt.show()



def main():
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/aptamer/dnase/apt_ms1_5.mzdata.xml_deconv_results.csv')
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/aptamer/dnase/apt_ms1_2_0_005.mzdata.xml_deconv_results.csv')
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/aptamer/dnase/apt_ms1_4_5_005.mzdata.xml_deconv_results.csv')
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/aptamer/dnase/apt_ms1_1_0_05.mzdata.xml_deconv_results.csv')
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/aptamer/010923/apt_ms1_5ul.mzdata.xml_deconv_results.csv')
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/aptamer/140923/apt10oe.mzdata.xml_deconv_results.csv')
    #data = pd.read_csv('/home/alex/Documents/LCMS/oligos/AlexR/ribo/ribo_138.mzdata.xml_deconv_results.csv')
    data = pd.read_csv('/home/alex/Documents/LCMS/oligos/AlexR/ribo/ribo_0109.mzdata.xml_deconv_results.csv')

    sorber = data_sorber(data)

    sorber.plot_data()

def main2():
    with open('sorber.obj', 'rb') as f:
        sorber = pickle.load(f)
    print(sorber.seq_data)
    print(sorber.masses)

    d = {}
    d['msses'] = sorber.masses
    d['rect'] = sorber.rect
    d['seq data'] = sorber.seq_data

    with open('sorber_extract_2.dict', 'wb') as f:
        pickle.dump(d, f)

def main3():
    fileName = '/home/alex/Documents/LCMS/oligos/aptamer/010923/apt_ms2_5ul_full_2.spectra'
    with open(fileName, 'rb') as f:
        data = pickle.load(f)

    for i, d in enumerate(data):
        print(i, d['mass'])

    spec_number = 18

    df = data[spec_number]['spectra']
    df['f.mass round'] = round(df['f.mass'], 0)
    group_df = df.groupby('f.mass round').agg({'f.mass': 'mean', 'f.intens': 'max'})

    spectra = ms2Spec_dataSorber(group_df, data[spec_number]['mass'], data[spec_number]['rt_interval'])

    #print(pd.DataFrame(spectra.interpr.ion_list))

    text = f"precursor mass: {round(data[spec_number]['mass'],1)} \n rt interval: {data[spec_number]['rt_interval']}\n"
    text += ('A C G T: ' +
             str(spectra.interpr.get_compose_by_mass(data[spec_number]['mass'], mass_type='base avg mass full', treshold=2)))
    spectra.title = text
    spectra.plot_data()

    #print(data[spec_number]['spectra'])
    #print(len(data[spec_number]['spectra']['f.mass'].unique()))
    #print(data[2])





if __name__ == '__main__':
    main()