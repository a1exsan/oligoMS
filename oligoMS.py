import numpy as np
from tqdm import tqdm
import pandas as pd
import oligoMass as omass
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
        #print([map[mz + i][t + k] for k in range(-1, 2)])
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

def neighbor_walk(mz, t, map, class_map, class_id):

    neighbor = []
    #true_map = [[0 for t in mz] for mz in map]

    p_mz, p_t = mz, t
    neighbor.append([p_mz, p_t, 1])
    class_map[p_mz][p_t] = class_id

    stop = False
    while not stop:

        for i in range(-1, 2):
            for j in range(-1, 2):
                #if not (i == 0 and j == 0):
                if abs(i) != abs(j):
                    if map[p_mz + i][p_t + j] > 0:
                        if class_map[p_mz + i][p_t + j] == 0:
                            class_map[p_mz + i][p_t + j] = class_id
                            neighbor.append([p_mz + i, p_t + j, 0])

        count = 0
        if len(neighbor) > 0:
            for i, n in enumerate(neighbor):
                if n[2] == 0:
                    p_mz, p_t = n[0], n[1]
                    neighbor[i][2] = 1
                    count += 1
                    break

        if count == 0:
            stop = True

    return np.array(neighbor), class_map

def find_clusters(data, map, param=3): #mod
    clusters = []

    class_map = [[0 for t in mz] for mz in map]
    class_id = 1
    for mz in tqdm(range(len(map)), desc='find clusters'):
        for t in range(len(map[0])):
            if map[mz][t] > 0 and class_map[mz][t] == 0:
                cluster, class_map = neighbor_walk(mz, t, map, class_map, class_id)
                class_id += 1
    for i, d in enumerate(data):
        mz, t = int(round(d[1]*param, 0)), int(round(d[0], 0))
        if class_map[mz][t] != 0:
            clusters.append([d[0], d[1], d[2], class_map[mz][t]])

    return np.array(clusters)

import time

def find_clusters_alg_2(data, low_intens_treshold=2000, crit_dist=2, rt_cf=3, number=1000):
    clusters = []

    df = pd.DataFrame(data, columns=['rt', 'mz', 'intens'])
    df = df[df['intens'] >= low_intens_treshold]
    df['class'] = np.zeros(df.shape[0])

    max_int = df['intens'].max()

    v = df[df['intens'] == max_int].values[0]
    v[3] = 1
    clusters.append(v)

    step_len = (max_int - low_intens_treshold) / number
    int_step = max_int

    pbar = tqdm(total=100, desc='finding clusters: ')
    iter_count = 0
    while int_step > low_intens_treshold:

        dff = df[df['intens'] < int_step]
        int_step -= step_len
        iter_count += step_len
        dff = dff[dff['intens'] >= int_step]

        if dff.shape[0] > 200:
            step_len = step_len / 2

        for i in range(dff.shape[0]):
            ctrl = True
            for k in range(len(clusters)):
                c = clusters[len(clusters) - k - 1]
                x, y = (dff.values[i][0] - c[0]) / rt_cf, dff.values[i][1] - c[1]
                dist = (x**2 + y**2)**0.5
                if dist <= crit_dist:
                    v = dff.values[i]
                    v[3] = clusters[len(clusters) - k - 1][3]
                    clusters.append(v)
                    ctrl = False
                    break
            if ctrl:
                v = dff.values[i]
                v[3] = np.array(clusters)[:, 3].max() + 1
                clusters.append(v)
        pbar.update(1)
    pbar.close()
    return np.array(clusters)

def filtrate_clusters(clust, dt_treshold):

    df = pd.DataFrame(clust, columns=['time', 'mz', 'intens', 'class'])
    clust_set = list(set(clust[:, 3]))
    results = np.array([df.loc[0]])
    #print(results)
    dt = df['time'].max() - df['time'].min()
    #dd = []
    for c in clust_set:
        cl = df[df['class'] == c]
        d = (cl['time'].max() - cl['time'].min()) * 100/ dt
        if d < dt_treshold:
            #print(cl.values)
            results = np.concatenate((results, cl.values), axis=0)
        #else:
        #    print(c, d)
        #dd.append(d)
    #plt.plot(range(len(dd)), dd, 'o')
    #plt.show()

    return results



def cluster_analysis(clust, top_number=3, negative_mode=True):

    results = []

    massed_clust = pd.DataFrame(clust, columns=['rt', 'mz', 'intens', 'class'])
    massed_clust['mass'] = np.zeros(clust.shape[0])
    #massed_clust = massed_clust.sort_values(by='class')

    clust_set = list(set(clust[:, 3]))

    df = pd.DataFrame(clust, columns=['time', 'mz', 'intens', 'class'])
    for c in tqdm(clust_set, desc='cluster analysis '):

        df_clust = df[df['class'] == c]

        other = pd.DataFrame(list(df_clust['mz']), columns=['group'])

        df_clust = df_clust.reset_index()
        df_clust = df_clust.join(other)
        df_clust = df_clust.round({'group':1})
        df_clust = df_clust.groupby('group').agg({'time': 'mean', 'intens': 'sum', 'class': 'first', 'mz': 'mean'})
        df_clust = df_clust.sort_values(by='intens', ascending=False)
        df_clust = df_clust.reset_index()
        #print(df_clust['class'].size, len(clust))
        #print(df_clust.loc[:5])
        df_sort = df_clust.loc[:top_number]
        df_sort = df_sort.sort_values(by='mz', ascending=True)
        mzs = list(df_sort['mz'])
        if len(mzs) > 1:
            s, count = 0., 0
            for i in range(len(mzs) - 1):
                    s += abs(mzs[i] - mzs[i + 1])
                    #print(abs(mzs[i] - mzs[i + 1]))
                    count += 1
            s = s / count
            charge = round(1 / s, 0)
            if negative_mode:
                sign = 1
            else:
                sign = -1
            mass = mzs[0] * charge + sign * charge
            d = {}
            d['mass'] = mass
            d['charge'] = charge
            d['mz'] = mzs[0]
            d['area'] = df_clust['intens'].sum()
            d['class'] = c
            d['rt'] = df_clust['time'].mean()
            results.append(d)

            massed_clust.loc[massed_clust['class'] == c, 'mass'] = mass

    return results, massed_clust

def group_mass_by_charge(mass_data, delta_mass=4, delta_rt=20):
    results = []

    ctrl = [0 for m in mass_data]

    area_sum = 0
    for i in range(len(mass_data)):
        if ctrl[i] == 0:
            d = {}
            d['mass'] = mass_data[i]['mass']
            d['area'] = mass_data[i]['area']
            d['rt'] = mass_data[i]['rt']
            d['charge'] = [mass_data[i]['charge']]
            d['class'] = mass_data[i]['class']
            count = 1
            ctrl[i] = 1
            mzi = round(mass_data[i]['mass'], 0)
            for j in range(len(mass_data)):
                mzj = round(mass_data[j]['mass'], 0)
                rti = mass_data[i]['rt']
                rtj = mass_data[j]['rt']

                if abs(mzi - mzj) <= delta_mass and i != j and abs(rti - rtj) <= delta_rt:
                    #print(mass_data[j]['mass'], mass_data[i]['mass'])
                    d['mass'] += mass_data[j]['mass']
                    d['area'] += mass_data[j]['area']
                    d['rt'] += mass_data[j]['rt']
                    d['charge'].append(mass_data[j]['charge'])
                    count += 1
                    ctrl[j] = 1

            if count > 0:
                #print(count, mzi)
                d['mass'] = d['mass'] / count
                d['rt'] = d['rt'] / count
            d['area%'] = d['area']
            area_sum += d['area']
            results.append(d)
    if area_sum > 0:
        for i in range(len(results)):
            results[i]['area%'] = round(results[i]['area%']*100 / area_sum, 2)
    return results

def data_rt_integration(np_data, low_treshold=5000):
    max_mz = np_data[:, 1].max()
    min_mz = np_data[:, 1].min()
    int_points = (round(max_mz, 0) - round(min_mz, 0)) * 10
    df = pd.DataFrame({'rt': np_data[:, 0], 'mz': np_data[:, 1], 'intens': np_data[:, 2]})
    dm = (max_mz - min_mz) / int_points
    m = min_mz + dm
    psum = 0

    out_data = []

    for i in tqdm(range(int(int_points)), desc='integration'):
        integ = df[df['mz'] <= m]['intens'].sum()
        if integ - psum > low_treshold:
            out_data.append([0., m - dm / 2, integ - psum])
        psum = integ
        m += dm

    return np.array(out_data)


class oligoMSanalysys():
    def __init__(self, fn, int_treshold=500, neighbor_treshold=60, max_mz=3200, rt_left=100):
        # путь к файлу с расширением mzML, файл с LCMS олигонуклеотида, записанный в MS1 режиме
        self.fn = fn
        # параметр фильтрации фона: int_treshold - верхняя граница отчечения фона
        self.int_treshold = int_treshold
        # параметр кластеризации: neighbor_treshold - нижний предел количества необходимых соседей в %, чтобы определить
        # принадлежность точки к кластеру.
        self.neighbor_treshold = neighbor_treshold
        # фильтр по массе
        self.max_mz = max_mz
        # левая граница по времени хроматограммы в сек.
        self.rt_left = rt_left
        # нижняя граница по интенсивности
        self.low_treshold = 1000
        self.scale_param = 4
        self.top_number = 3
        self.delta_mass = 4
        self.delta_rt = 20
        self.dt_treshold = 25
        self.critic_distance = 1.1
        self.rt_coeff = 4
        self.points_number = 1000

        self.mass_tab = None

    def ms_analysis(self):
        data, bkg = open_mzml(self.fn, rt_left=self.rt_left)

        data = substract_bkg(data, bkg, treshold=self.int_treshold)

        for i in range(3):
            map = get_intensity_map(data, low_treshold=self.low_treshold, param=self.scale_param)
            data = find_inner_points(data, map, neighbor_treshold=self.neighbor_treshold, param=self.scale_param)


        clust = find_clusters_alg_2(data, low_intens_treshold=self.low_treshold
                                        , crit_dist=self.critic_distance
                                        , rt_cf=self.rt_coeff, number=self.points_number)

        clust = filtrate_clusters(clust, dt_treshold=self.dt_treshold)

        cluster_tab, self.massed_clust = cluster_analysis(clust, top_number=self.top_number)

        mass_res = group_mass_by_charge(cluster_tab, delta_mass=self.delta_mass, delta_rt=self.delta_rt)

        self.mass_tab = pd.DataFrame(mass_res)

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
        data['charge'] = np.ones(data['mz'].shape[0])
        data['mass'] = np.ones(data['mz'].shape[0])

        if self.is_positive:
            sign = -1
        else:
            sign = 1

        for cl in classes:
            df = data[data['class'] == cl]
            if df.shape[0] > 1:
                df = df.sort_values(by='mz', ascending=False)
                charge = round(1 / abs(df['mz'].values[0] - df['mz'].values[1]), 0)

                r_int = df['intens'] / df['intens'].sum()
                masses = df['mz'] * charge + sign * charge
                avg_mass = (masses * r_int).sum()

                data.loc[data['class'] == cl, 'charge'] = charge
                data.loc[data['class'] == cl, 'mass'] = avg_mass

        data['mono_mass'] = data['mz'] * data['charge'] + sign * data['charge']

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
        #print(data)
        #print(df)
        return data.drop(list(df.index))

class MassExplainer():
    def __init__(self, seq, mass_tab):
        self.mass_tab = mass_tab
        self.seq = seq
        self.generate_hypothesis()
        dna = osa.dnaSeq(seq)
        self.molecular_weight = dna.getMolMass()

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
            dna = osa.dnaSeq(h['seq'])
            molecular_weight = dna.getMolMass()
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
            dna = osa.dnaSeq(h['seq'])
            molecular_weight = dna.getMolMass()
            for i, m in enumerate(massTab):
                if abs(m['mass'] - molecular_weight * h['cf'] - h['deltaM']) <= mass_treshold:
                    massTab[i]['type'] = h['type']
                    massTab[i]['name'] = h['name']
                    massTab[i]['seq'] = h['seq']

        self.mass_tab = pd.DataFrame(massTab)
        self.mass_tab = self.mass_tab.fillna('unknown')
        self.mass_tab['area'] = np.zeros(self.mass_tab.shape[0])
        total = self.mass_tab['intens'].sum()

        #names = list(set(self.mass_tab['name']))
        #for name in names:
        #    area = self.mass_tab[self.mass_tab['name'] == name]['intens'].sum() * 100 / total
        #    self.mass_tab.loc[self.mass_tab['name'] == name, 'area'] = area

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
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = 'main', self.seq, 0., 'main', 1
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = 'main +Na', self.seq, 23, 'main', 1
        self.hypo_tab.append(d)

        d = {}
        d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = 'main Dimer', self.seq, 0., 'main', 2
        self.hypo_tab.append(d)

        dna = osa.dnaSeq(self.seq)

        for i in range(1, len(dna.string2seq(self.seq)) - 2):
            d = {}
            dna = osa.dnaSeq(self.seq)
            seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = f'5 end n - {i}', seq, 0., f'5 end n - {i}', 1
            self.hypo_tab.append(d)

            d = {}
            dna = osa.dnaSeq(self.seq)
            seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = f'5 end n - {i} Dimer', seq, 0., f'5 end n - {i}', 2
            self.hypo_tab.append(d)

            d = {}
            dna = osa.dnaSeq(self.seq)
            seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="5'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = f'5 end n - {i} +Na', seq, 23., f'5 end n - {i}', 1
            self.hypo_tab.append(d)

            d = {}
            dna = osa.dnaSeq(self.seq)
            seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = f'3 end n - {i}', seq, 0., f'3 end n - {i}', 1
            self.hypo_tab.append(d)

            d = {}
            dna = osa.dnaSeq(self.seq)
            seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = f'3 end n - {i} Dimer', seq, 0., f'3 end n - {i}', 2
            self.hypo_tab.append(d)

            d = {}
            dna = osa.dnaSeq(self.seq)
            seq = dna.seq_end_cut(self.seq, cut_number=i, end_type="3'")
            d['name'], d['seq'], d['deltaM'], d['type'], d['cf'] = f'3 end n - {i} +Na', seq, 23., f'3 end n - {i}', 1
            self.hypo_tab.append(d)

    def filtrate_mass_tab(self, massed_clust, treshold=0.5):
        ctrl = True
        if ctrl:
            self.mass_tab['rt_ctrl'] = np.ones(self.mass_tab.shape[0])
            mass_main = self.mass_tab[self.mass_tab['name'] == 'main']['mass'].values[0]
            rt_main = self.mass_tab[self.mass_tab['name'] == 'main']['rt'].values[0]
            mass_main -= 1000
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


def main():
    path = r'C:\Users\Alex\Documents\LCMS\Oligos\NR_des_ms2.mgf'
    #path = r'C:\Users\Alex\Documents\LCMS\Oligos\NR_des_ms2_15cid.mgf'

    mgf = nms.MSspecData()
    mgf.loadMGF(path)

    x, y = [], []
    for t in tqdm(mgf.specTab, desc='Searching'):
        if abs(float(t['mz']) - 1740) <= 1.2:
            print(max(t['intens']), t['ms2'][t['intens'].index(max(t['intens']))], t['charge'], t['rt'])

            if max(t['intens']) == 10675.62:
            #if max(t['intens']) == 94200.73:
                draw_ms2(t['ms2'], t['intens'])

            if abs(t['ms2'][t['intens'].index(max(t['intens']))] - 1946) <= 1.2:
                x.append(t['rt'])
                y.append(max(t['intens']))

    #plt.plot(x, y, '-')
    #plt.show()


    #modChrom = nms.rnaModFinder(mgf.specTab)

    #viewer = nms.gaussChromViewer(modChrom, path)
    #viewer.draw_crom_tab()

def main1():
    #data, bkg = open_mzml(r'C:\Users\Alex\PycharmProjects\lcms_utils\data\TOPPAS_out\003-FileConverter-out\NR_des.mzML')
    #data, bkg = open_mzml(r'C:\Users\Alex\Documents\LCMS\peptides\Dima\init_peptides\21w047-8_5ul.mzML')
    data, bkg = open_mzml(r'C:\Users\Alex\Documents\LCMS\peptides\Dima\init_peptides\151221\111.mzML', rt_left=600)

    #plt.scatter(data[:, 0], data[:, 1], color='red', s=8)
    #plt.show()

    data = substract_bkg(data, bkg, treshold=5000)

    #plt.scatter(data[:, 0], data[:, 1], color='red', s=8)
    #plt.show()

    map = get_intensity_map(data)

    data = find_inner_points(data, map, neighbor_treshold=80)

    map = get_intensity_map(data)

    clust = find_clusters(data, map)

    clust = filtrate_clusters(clust, dt_treshold=20)

    res = cluster_analysis(clust, negative_mode=False)

    mass_res = group_mass_by_charge(res)

    df_res = pd.DataFrame(mass_res)
    df_res = df_res.sort_values(by='area', ascending=False)
    #print(df_res)

    for i, j, k, c in zip(df_res['mass'].loc[:], df_res['area%'].loc[:], df_res['rt'].loc[:], df_res['charge'].loc[:]):
        print(i, j, k, c)

    #for r in res:
    #    print(r)

    plt.scatter(clust[:, 0], clust[:, 1], color='red', s=8)

    print(set(clust[:, 3]))
    df = pd.DataFrame(clust, columns=['time', 'mz', 'intens', 'class'])
    df = df[df['class'] == 12.0]

    plt.scatter(df['time'], df['mz'], color='blue', s=2)
    plt.show()

def test_deconv():
    name = 's10'
    #data, bkg = open_mzml(rf'C:\Users\Alex\Documents\LCMS\Oligos\Nikolay\{name}.mzML', rt_left=100)
    data, bkg = open_mzml_2(rf'/home/alex/Documents/LCMS/oligos/Nikolay/210122/{name}.mzML', rt_left=100)
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