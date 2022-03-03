import numpy as np
from tqdm import tqdm
import pandas as pd
from oligoMass import dna as omass
from oligoMass import molmassOligo as mmo
import pickle
import msvis
import pymzml

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def open_mzml(fn, int_treshold=5000, max_mz=3200, rt_left=100):

    exp = pymzml.run.Reader(fn)

    vec = [0 for i in range(int(round(max_mz, 0)))]

    data = []
    for s in tqdm(exp):
        rt = s.scan_time[0]
        if rt >= rt_left:
            for p in s.peaks("centroided"):
                if p[1] >= int_treshold:
                    v = [rt, p[0], p[1]]
                    data.append(v)
                    vec[int(round(p[0], 0))] += 1

    return np.array(data), vec

def substract_bkg(data, bkg, treshold=3000):
    ret = []

    mz_list = [i for i, f in enumerate(bkg) if f >= treshold]

    for d in tqdm(data):
        if not (int(round(d[1], 0)) in mz_list):
            ret.append(d)

    return np.array(ret)

def get_intensity_map(data, low_treshold=1000, param=3): #mod
    max_mz = int(round(max(data[:, 1])*param, 0))
    max_t = int(round(max(data[:, 0]), 0))
    out = [[0. for t in range(0, max_t + 10)] for mz in range(0, max_mz + 10)]

    for d in data:
        if d[2] >= low_treshold:
            mz, t = int(round(d[1]*param, 0)), int(round(d[0], 0))
            out[mz][t] = d[2]
    return out

def find_neighbor(mz, t, map, param=3): #mod
    count = 0
    mz, t = int(round(mz*param, 0)), int(round(t, 0))
    for i in range(-1, 2):
        for j in range(-1, 2):
            if map[mz + i][t + j] > 0:
                count += 1
    return count * 100 / 9

def find_inner_points(data, map, neighbor_treshold=60, param=3):
    out = []

    for d in data:
        if find_neighbor(d[1], d[0], map, param=param) >= neighbor_treshold:
            out.append(d)

    return np.array(out)

class mzSpecDeconv():
    def __init__(self, mz_array, int_array, is_positive=False):
        self.data = pd.DataFrame({'mz': mz_array, 'intens': int_array,
                                  'class': np.zeros(mz_array.shape[0]),
                                  'mass': np.zeros(mz_array.shape[0])})
        self.is_positive = is_positive

    def __clusterize(self, data):

        clusters = []
        clusters.append([list(data.loc[0])])
        for index in range(1, data.shape[0]):
            finded = False
            for cl_index in range(len(clusters)):
                for mz_index in range(len(clusters[len(clusters) - cl_index - 1])):
                    mz_cl = clusters[len(clusters) - cl_index - 1][mz_index][0]
                    mz = data['mz'].loc[index]
                    if abs(mz - mz_cl) <= 1:
                        finded = True
                        clusters[len(clusters) - cl_index - 1].append(list(data.loc[index]))
                        data['class'].loc[index] = len(clusters) - cl_index - 1
                        break
                if finded:
                    break
            if not finded:
                clusters.append([list(data.loc[index])])
                data['class'].loc[index] = len(clusters) - 1

        return data

    def __compute_mass(self, data):

        classes = list(set(data['class']))
        data['charge'] = np.zeros(data['mz'].shape[0])
        data['mass'] = np.ones(data['mz'].shape[0])

        if self.is_positive:
            sign = -1
        else:
            sign = 1

        for cl in classes:
            df = data[data['class'] == cl]
            if df.shape[0] > 3:
                df = df.sort_values(by='mz', ascending=False)
                #charge = round(1 / abs(df['mz'].values[0] - df['mz'].values[1]), 0)

                diff = pd.DataFrame(df['mz'])
                diff['diff'] = df['mz'].diff(periods=1)
                diff.dropna(inplace=True)
                diff = diff[diff['diff'] != 0]
                diff['charge'] = [abs(round(1/z, 0)) for z in diff['diff']]
                charge = diff['charge'].value_counts().idxmax()

                r_int = df['intens'] / df['intens'].sum()
                masses = df['mz'] * charge + sign * charge
                avg_mass = (masses * r_int).sum()

                data.loc[data['class'] == cl, 'charge'] = charge
                data.loc[data['class'] == cl, 'mass'] = avg_mass

        data['mono_mass'] = data['mz'] * data['charge'] + sign * data['charge']
        data = data[data['charge'] > 0]

        return data


    def deconvolute(self):
        self._data = self.data.sort_values(by='intens', ascending=False)
        self._data = self._data.reset_index()
        self._data = self._data.drop(['index'], axis=1)

        self._data = self.__clusterize(self._data)
        self._data = self.__compute_mass(self._data)

        #print(self._data)
        return self._data

    @staticmethod
    def drop_by_charge(data, max_charge=10):
        data = data[data['charge'] <= max_charge]
        return data

class oligosDeconvolution():
    def __init__(self, rt, mz, intens, is_positive=False, max_charge=10):
        rt = [round(i, 0) for i in rt]
        self.data = pd.DataFrame({'rt': rt, 'mz': mz, 'intens': intens})
        self.is_positive = is_positive
        self.max_charge = max_charge

    def deconvolute(self):
        rt_list = list(set(self.data['rt']))
        sum_data = np.array([])
        for rt in tqdm(rt_list, desc='Deconvolution:'):
            df = self.data[self.data['rt'] == rt]
            deconv = mzSpecDeconv(df['mz'], df['intens'], is_positive=self.is_positive)
            data = deconv.deconvolute()
            data = deconv.drop_by_charge(data, self.max_charge)
            data['rt'] = rt * np.ones(data['mz'].shape[0])

            if sum_data.shape[0] == 0:
                sum_data = data
            else:
                if data.shape[0] > 0:
                    sum_data = pd.concat([sum_data, data])

        sum_data = sum_data.sort_values(by='intens', ascending=False)
        sum_data = sum_data.reset_index()
        sum_data = sum_data.drop(['index'], axis=1)
        return sum_data

    @staticmethod
    def drop_data(data, mass_max, mass_min, rt_min, rt_max):
        df = data[data['mass'] <= mass_max]
        df = df[df['mass'] >= mass_min]
        df = df[df['rt'] <= rt_max]
        df = df[df['rt'] >= rt_min]
        return data.drop(list(df.index))

    @staticmethod
    def rt_filtration(data, rt_min, rt_max):
        df = data[data['rt'] <= rt_max]
        df = df[df['rt'] >= rt_min]
        return df

class MassExplainer():
    def __init__(self, seq, mass_tab):
        self.mass_tab = mass_tab
        self.seq = seq
        self.generate_hypothesis()
        #dna = omass.oligoSeq(seq)
        #self.molecular_weight = dna.getMolMass()
        dna = mmo.oligoNASequence(seq)
        self.molecular_weight = dna.getAvgMass()

    def generate_hypothesis(self):
        self.hypo_tab = []

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = 'main', self.seq, 0., 'main', 1
        self.hypo_tab.append(d)

    def group_by_type(self):
        self.gTab = self.mass_tab.groupby('type').agg(
            {'mass':'min', 'area':'sum', 'rt':'mean',
            'charge':'max', 'class':'max', 'area%':'sum',
            'type':'first', 'name':'first', 'seq':'first'})

        self.gTab['purity%'] = (self.gTab['area'] / self.gTab['area'].sum()) * 100

        self.gTab = self.gTab.reset_index(drop=True)

    def group_by_type_2(self):
        self.gTab = self.mass_tab.groupby('type').agg(
            {'mass':'first', 'rt':'mean',
            'charge':'max', 'class':'max', 'area%':'sum',
            'type':'first', 'name':'first', 'seq':'first'})

        self.gTab['purity%'] = self.gTab['area%']

        self.gTab = self.gTab.reset_index(drop=True)

    def explain(self, mass_treshold=3):

        massTab = list(self.mass_tab.T.to_dict().values())
        for h in self.hypo_tab:
            #dna = omass.oligoSeq(h['seq'])
            dna = mmo.oligoNASequence(h['seq'])
            #molecular_weight = dna.getMolMass()
            molecular_weight = dna.getAvgMass()
            for i, m in enumerate(massTab):
                if abs(m['mass'] - molecular_weight * h['cf'] - h['deltaM']) <= mass_treshold:
                    massTab[i]['type'] = h['type']
                    massTab[i]['name'] = h['name']
                    massTab[i]['seq'] = h['seq']

        self.mass_tab = pd.DataFrame(massTab)
        self.mass_tab = self.mass_tab.sort_values(by='area', ascending=False)
        self.mass_tab = self.mass_tab.fillna('unknown')

    def explain_2(self, mass_treshold=3):

        massTab = list(self.mass_tab.T.to_dict().values())
        for h in self.hypo_tab:
            #dna = omass.oligoSeq(h['seq'])
            dna = mmo.oligoNASequence(h['seq'])
            #molecular_weight = dna.getMolMass()
            molecular_weight = dna.getAvgMass()
            for i, m in enumerate(massTab):
                if abs(m['mass'] - molecular_weight * h['cf'] - h['deltaM']) <= mass_treshold:
                    massTab[i]['type'] = h['type']
                    massTab[i]['name'] = h['name']
                    massTab[i]['seq'] = h['seq']

        self.mass_tab = pd.DataFrame(massTab)
        self.mass_tab = self.mass_tab.fillna('unknown')
        self.mass_tab['area'] = np.zeros(self.mass_tab.shape[0])
        total = self.mass_tab['intens'].sum()

        self.mass_tab['area%'] = self.mass_tab['intens'] * 100 / total
        self.mass_tab = self.mass_tab.sort_values(by='area', ascending=False)

    @staticmethod
    def drop_unknown(data):
        df = data[data['name'] == 'unknown']
        data = data.drop(list(df.index))

        total = data['intens'].sum()
        names = list(set(data['name']))
        for name in names:
            area = data[data['name'] == name]['intens'].sum() * 100 / total
            data.loc[data['name'] == name, 'area'] = area
        data = data.sort_values(by='area', ascending=False)
        #print(data['area'].sum())
        #print(data)
        return data


class oligoMassExplainer(MassExplainer):
    def __init__(self, seq, mass_tab):
        super().__init__(seq, mass_tab)

    def generate_hypothesis(self):
        self.hypo_tab = []

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'main', self.seq, 0., 'main', 1, 2
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'main +Na', self.seq, 23, 'main', 1, 3
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'main Dimer', self.seq, 0., 'main', 2, 2
        self.hypo_tab.append(d)

        #dna = omass.oligoSeq(self.seq)
        dna = mmo.oligoNASequence(self.seq)

        for i in range(1, dna.size()):
            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            seq = dna.getPrefix(i).sequence
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'5 end n - {dna.size() - i}', seq, 0., f'5 end n - {dna.size() - i}', 1, 2
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'5 end n - {dna.size() - i} Dimer', seq, 0., f'5 end n - {dna.size() - i}', 2, 2
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'5 end n - {dna.size() - i} +Na', seq, 23., f'5 end n - {dna.size() - i}', 1, 3
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            seq = dna.getSuffix(i).sequence
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'3 end n - {i}', seq, 0., f'3 end n - {i}', 1, 2
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'3 end n - {i} Dimer', seq, 0., f'3 end n - {i}', 2, 2
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'3 end n - {i} +Na', seq, 23., f'3 end n - {i}', 1, 3
            self.hypo_tab.append(d)

    def filtrate_mass_tab(self, massed_clust, treshold=0.5, mass_thold=500):
        ctrl = True
        if ctrl:
            self.mass_tab['rt_ctrl'] = np.ones(self.mass_tab.shape[0])
            mass_main = self.mass_tab[self.mass_tab['name'] == 'main']['mass'].values[0]
            rt_main = self.mass_tab[self.mass_tab['name'] == 'main']['rt'].values[0]
            mass_main -= mass_thold
            rt_main -= 20
            df = self.mass_tab[self.mass_tab['mass'] <= mass_main]
            df = df[df['rt'] >= rt_main]
            df = df[df['name'] != 'main']

            for index in df.index:
                self.mass_tab['rt_ctrl'].loc[index] = 0
            self.mass_tab = self.mass_tab[self.mass_tab['rt_ctrl'] == 1]

        self.mass_tab = self.mass_tab[self.mass_tab['area%'] >= treshold]
        self.mass_tab['area%'] = (self.mass_tab['area'] / self.mass_tab['area'].sum()) * 100
        return massed_clust

    def labeling_mass_tab(self, massed_clust):
        massed_clust['name'] = ['unknown' for i in range(massed_clust.shape[0])]

        for name, clss in zip(self.mass_tab['type'], self.mass_tab['class']):
            if name != 'unknown':
                massed_clust.loc[massed_clust['class'] == clss, 'name'] = name
        return massed_clust

    def drop_artifacts(self, mass_thold=500):
        df = self.mass_tab[self.mass_tab['name'] != 'unknown']
        for name in list(set(df['name'])):
            if name.find('Dimer') == -1:
                df = self.mass_tab[self.mass_tab['name'] == name]
                max_mass = df['mass'].max() - mass_thold
                rt_min = df['rt'].min()
                rt_max = df['rt'].max()
                self.mass_tab = oligosDeconvolution.drop_data(self.mass_tab, max_mass, 0, rt_min, rt_max)

class oligoMassExplainer2(oligoMassExplainer):
    def __init__(self, seq, mass_tab):
        super().__init__(seq, mass_tab)

    def generate_hypothesis(self):
        self.hypo_tab = []

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'main', self.seq, 0., 'main', 1, 2
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'main +Na', self.seq, 23, 'main', 1, 3
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'main Dimer', self.seq, 0., 'main', 2, 2
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'like main', self.seq, 0., 'like main', 1, 100
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'aPurin_A', self.seq, 135., 'aPurin_A', 1, 7
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
            'aPurin_G', self.seq, 150., 'aPurin_G', 1, 7
        self.hypo_tab.append(d)

        #dna = omass.oligoSeq(self.seq)
        dna = mmo.oligoNASequence(self.seq)

        #for i in range(1, len(dna.string2seq(self.seq)) - 2):
        for i in range(1, dna.size()):

            if i > 1:
                d = {}
                seq = dna.getDeletion(i)
                d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                    f'Del - {i}', seq, 0., f'Del - {i}', 1, 2
                self.hypo_tab.append(d)


            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            seq = dna.getPrefix(i).sequence
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'5 end n - {dna.size() - i}', seq, 0., f'5 end n - {dna.size() - i}', 1, 2
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'like 5 end n - {dna.size() - i}', seq, 0., f'like 5 end n - {dna.size() - i}', 1, 150
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'5 end n - {dna.size() - i} +Na', seq, 23., f'5 end n - {dna.size() - i}', 1, 3
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            seq = dna.getSuffix(i).sequence
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'3 end n - {i}', seq, 0., f'3 end n - {i}', 1, 2
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'like 3 end n - {i}', seq, 0., f'like 3 end n - {i}', 1, 100
            self.hypo_tab.append(d)

            d = {}
            #dna = omass.oligoSeq(self.seq)
            #seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'], d['thold'] = \
                f'3 end n - {i} +Na', seq, 23., f'3 end n - {i}', 1, 3
            self.hypo_tab.append(d)

    def explain_2(self, mass_treshold=3):

        massTab = list(self.mass_tab.T.to_dict().values())
        for h in self.hypo_tab:
            if h['name'].find('like') == -1:
                #dna = omass.oligoSeq(h['seq'])
                #molecular_weight = dna.getMolMass()
                dna = mmo.oligoNASequence(h['seq'])
                molecular_weight = dna.getAvgMass()
                mass_treshold = h['thold']
                if h['name'].find('aPurin') == -1:
                    for i, m in enumerate(massTab):
                        if abs(m['mass'] - molecular_weight * h['cf'] - h['deltaM']) <= mass_treshold:
                            massTab[i]['type'] = h['type']
                            massTab[i]['name'] = h['name']
                            massTab[i]['seq'] = h['seq']
                else:
                    for i, m in enumerate(massTab):
                        diff = molecular_weight * h['cf'] - h['deltaM']
                        if abs(m['mass'] - diff) <= mass_treshold and diff > 0:
                            massTab[i]['type'] = h['type']
                            massTab[i]['name'] = h['name']
                            massTab[i]['seq'] = h['seq']

        self.mass_tab = pd.DataFrame(massTab)
        self.mass_tab = self.mass_tab.fillna('unknown')

        massTab = list(self.mass_tab.T.to_dict().values())
        for h in self.hypo_tab:
            if h['name'].find('like') != -1:
                #dna = omass.oligoSeq(h['seq'])
                #molecular_weight = dna.getMolMass()
                dna = mmo.oligoNASequence(h['seq'])
                molecular_weight = dna.getAvgMass()
                for i, m in enumerate(massTab):
                    if m['name'] == 'unknown':
                        #print(h['seq'], h['name'])
                        mass_treshold = h['thold']
                        diff = m['mass'] - molecular_weight * h['cf'] - h['deltaM']
                        if abs(diff) <= mass_treshold:
                            massTab[i]['type'] = h['type']
                            massTab[i]['name'] = h['name']
                            massTab[i]['seq'] = h['seq']

        self.mass_tab = pd.DataFrame(massTab)

        self.mass_tab['area'] = np.zeros(self.mass_tab.shape[0])
        total = self.mass_tab['intens'].sum()

        self.mass_tab['area%'] = self.mass_tab['intens'] * 100 / total
        self.mass_tab = self.mass_tab.sort_values(by='area', ascending=False)



def test_deconv():
    name = 's10'
    data, bkg = open_mzml(rf'/home/alex/Documents/LCMS/oligos/Nikolay/210122/{name}.mzML', rt_left=100)
    data = substract_bkg(data, bkg, treshold=600)
    for i in range(3):
        map = get_intensity_map(data, low_treshold=1000, param=4)
        data = find_inner_points(data, map, neighbor_treshold=60, param=4)

    oligoD = oligosDeconvolution(data[:, 0], data[:, 1], data[:, 2])
    data_ = oligoD.deconvolute()

    print(data_)

    #df = pd.DataFrame({'rt': [round(i) for i in data[:, 0]],
    #                   'mz': data[:, 1], 'intens': data[:, 2]})

    rt_pos = 1064
    #df = df[df['rt'] == rt_pos]

    #deconv = mzSpecDeconv(df['mz'], df['intens'])
    #data_ = deconv.deconvolute()

    #print(data_.shape[0], len(data[:, 0]))
    spec_viewer = msvis.bokeh_ms_map(data_['rt'],
                                         data_['mass'], data_['intens'], rt_position=rt_pos)
    spec_viewer.draw_map(is_show=True)


if __name__=='__main__':

    test_deconv()