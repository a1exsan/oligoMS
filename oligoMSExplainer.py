
import pandas as pd
from oligoMass import molmassOligo as mmo
import mzdatapy as mzdata
import numpy as np

class chromInteg():
    def __init__(self, TIC, baseline=None, baseconst=100):
        if baseline == None:
            self.baseline = np.zeros(TIC.shape[0]) + baseconst
        else:
            self.baseline = baseline

        self.TIC = TIC
        self.substruct_bkg()

    def substruct_bkg(self):
        self.TIC -= self.baseline
        self.TIC = np.array([i if i > 0 else 0 for i in self.TIC])

    def integrate(self, a, b):
        target = sum(self.TIC[a : b])
        return target * 100 / sum(self.TIC)

class lcmsData():
    def __init__(self, fn='', mode='-', int_treshold=5000, init_data=None, bkg=None):
        self.fn = fn
        self.mode = mode
        self.max_rt = 2000
        self.min_rt = 0
        self.min_mz = 0
        self.int_treshold = int_treshold

        if fn != '':
            self.open_file()
        else:
            self.data = init_data.copy()
            self.bkg = bkg.copy()
            self.mz_max = np.max(self.data[:, 1])

    def open_file(self):
        spec = mzdata.mzdata(self.fn)
        self.data, self.bkg = spec.mzdata2tab_all(int_treshold=self.int_treshold)
        self.mz_max = np.max(self.data[:, 1])

    def substruct_bkg(self, min_rt=0, max_rt=5000, min_mz=0, treshold=500, int_treshold=5000):
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_mz = min_mz
        self.int_treshold = int_treshold

        ret = []

        mz_list = [i for i, f in enumerate(self.bkg) if f >= treshold]

        for d in self.data:
            if not (int(round(d[1], 0)) in mz_list):
                if d[0] >= min_rt and d[0] <= max_rt and d[1] >= min_mz and d[1] <= self.mz_max:
                    if d[2] >= self.int_treshold:
                        ret.append(d)

        self.data = np.array(ret)

    def ms2matrix_1(self):

        self.rt_max = np.max(self.data[:, 0])
        self.mz_max = np.max(self.data[:, 1])

        mz_shape = int(round((self.mz_max + 1), 0)) + 1
        rt_shape = int(round((self.rt_max + 1), 0)) + 1

        self.matrix = np.zeros((mz_shape, rt_shape))
        self.bit_matrix = np.zeros((mz_shape, rt_shape))

        for d in self.data:
            rt = int(round(d[0]))
            mz = int(round(d[1]))
            self.matrix[mz, rt] += d[2]
            if d[2] > 0:
                self.bit_matrix[mz, rt] = 1

    def vector_shape(self):
        return self.matrix[:, 0].shape[0]


class oligoLeaf():
    def __init__(self, seq='', ltype='init', lname='init', mass_min=0, mass_max=0, multimer_num=1):
        self.seq = seq
        self.type = ltype
        self.name = lname
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.delta_mass = mass_max - mass_min
        self.multimer_num = multimer_num
        self.set_mass()

    def set_mass(self):
        if self.seq == '':
            self.mass = 0.
            self.mono_mass = 0.
        else:
            oligo = mmo.oligoNASequence(sequence=self.seq)
            self.mass = oligo.getAvgMass() * self.multimer_num
            self.mono_mass = oligo.getMonoMass() * self.multimer_num

    def get_isotop_dist(self, charge):
        mono = int(round(self.mono_mass, 0))
        mass = int(round(self.mass, 0))
        if self.delta_mass == 0:
            count = (mass - mono) * 2 + 1
            m = mono
        else:
            count = int(self.delta_mass)
            m = mono + self.mass_min

        out = {'charge': [], 'mz': [], 'mz int': []}
        mz_init = 0.
        for i in range(count):
            mz = (m - charge) / charge
            if round(mz) > round(mz_init):
                out['charge'].append(charge)
                out['mz'].append(mz)
                out['mz int'].append(int(round(mz, 0)))
            m += 1
            mz_init = mz
        return out

    def get_mz_forms(self, min_mz=500, max_mz=3200):
        self.forms = {'charge': [], 'mz': [], 'mz int': []}
        charge = 1
        while True:
            mz = (self.mass - charge) / charge
            isotop_dist = self.get_isotop_dist(charge)
            if mz + self.mass_min >= min_mz and mz + self.mass_max <= max_mz:
                self.forms['charge'].extend(isotop_dist['charge'])
                self.forms['mz'].extend(isotop_dist['mz'])
                self.forms['mz int'].extend(isotop_dist['mz int'])
            elif mz + self.mass_min < min_mz:
                break
            charge += 1
        return self.forms

    def set_vector(self, shape):
        self.vector = np.zeros(shape)
        self.negative = np.ones(shape)
        for mz in self.forms['mz int']:
            if mz < self.vector.shape[0]:
                self.vector[mz] = 1.
                self.negative[mz] = 0.


class oligoTree():
    def __init__(self, sequence, vector_shape, min_mz=500, max_mz=3200,
                 score_treshold=0.5):
        self.seq = sequence
        self.vector_shape = vector_shape
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.score_treshold = score_treshold
        self.__oligos = []
        self.__set_forms()

    def __set_forms(self):
        self.__main()
        self.__5_3_deletion_n_x()
        self.__main_multimer()

    def __main(self):
        oligo = oligoLeaf(f'{self.seq}', ltype='main', lname='main', mass_min=0, mass_max=0, multimer_num=1)
        oligo.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
        oligo.set_vector(shape=self.vector_shape)
        self.__oligos.append(oligo)

        oligo = oligoLeaf(f'[Na]{self.seq}', ltype='main', lname='main+Na adducts', mass_min=0, mass_max=0, multimer_num=1)
        oligo.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
        oligo.set_vector(shape=self.vector_shape)
        self.__oligos.append(oligo)

        oligo = oligoLeaf(f'{self.seq}', ltype='main', lname='main adducts', mass_min=10, mass_max=140,
                          multimer_num=1)
        oligo.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
        oligo.set_vector(shape=self.vector_shape)
        self.__oligos.append(oligo)

    def __main_multimer(self):
        oligo = oligoLeaf(f'{self.seq}', ltype='main', lname='main dimer', mass_min=0, mass_max=0, multimer_num=2)
        oligo.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
        oligo.set_vector(shape=self.vector_shape)
        self.__oligos.append(oligo)

        oligo = oligoLeaf(f'{self.seq}', ltype='main', lname='main trimer', mass_min=0, mass_max=0, multimer_num=3)
        oligo.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
        oligo.set_vector(shape=self.vector_shape)
        self.__oligos.append(oligo)

        oligo = oligoLeaf(f'{self.seq}', ltype='main', lname='main tetramer', mass_min=0, mass_max=0, multimer_num=4)
        oligo.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
        oligo.set_vector(shape=self.vector_shape)
        self.__oligos.append(oligo)

    def __5_3_deletion_n_x(self):
        oligo = mmo.oligoNASequence(self.seq)
        N = oligo.getSeqLength()
        for i in range(1, oligo.getSeqLength()):

            prefix = oligo.getPrefix(i)
            suffix = oligo.getSuffix(i)
            if i > 1:
                deletion = oligo.getDeletion(i)
            else:
                deletion = ''

            oligo1 = oligoLeaf(f'{prefix.sequence}', ltype=f'3n-{N - i}', lname=f'3n-{N - i}', mass_min=0, mass_max=0,
                              multimer_num=1)
            oligo1.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
            oligo1.set_vector(shape=self.vector_shape)
            self.__oligos.append(oligo1)

            oligo1 = oligoLeaf(f'{prefix.sequence}', ltype=f'3n-{N - i}', lname=f'3n-{N - i} adducts', mass_min=10, mass_max=120,
                               multimer_num=1)
            oligo1.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
            oligo1.set_vector(shape=self.vector_shape)
            self.__oligos.append(oligo1)

            oligo1 = oligoLeaf(f'{suffix.sequence}', ltype=f'5n-{i}', lname=f'5n-{i}', mass_min=0, mass_max=0,
                              multimer_num=1)
            oligo1.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
            oligo1.set_vector(shape=self.vector_shape)
            self.__oligos.append(oligo1)

            oligo1 = oligoLeaf(f'{suffix.sequence}', ltype=f'5n-{i}', lname=f'5n-{i} adducts', mass_min=10, mass_max=120,
                               multimer_num=1)
            oligo1.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
            oligo1.set_vector(shape=self.vector_shape)
            self.__oligos.append(oligo1)

            oligo1 = oligoLeaf(f'{deletion}', ltype=f'del', lname=f'del-{i}', mass_min=0, mass_max=0,
                              multimer_num=1)
            oligo1.get_mz_forms(min_mz=self.min_mz, max_mz=self.max_mz)
            oligo1.set_vector(shape=self.vector_shape)
            self.__oligos.append(oligo1)
            #print(prefix.sequence, suffix.sequence, deletion)

    def mul(self, lcms_data):
        ret = {'data': [], 'chrom': [], 'MSscore': [], 'score chrom': [], 'bit score': [], 'rt index': []}
        matrix = lcms_data.matrix.T
        bit_matrix = lcms_data.bit_matrix.T
        #total_intens_vec = np.sum(matrix, axis=1) + 1.
        for oligo in self.__oligos:
            vec = matrix @ oligo.vector
            score_vec = bit_matrix @ oligo.vector
            ret['chrom'].append(vec)
            #ret['score chrom'].append(vec / total_intens_vec)
            sum_vec = np.sum(oligo.vector)
            if sum_vec != 0:
                ret['score chrom'].append(score_vec / sum_vec)
                ret['bit score'].append(np.array([1 if i >= self.score_treshold else 0 for i in ret['score chrom'][-1]]))
                rt_index = np.array([i for i in range(ret['score chrom'][-1].shape[0])]) * ret['bit score'][-1]
                ret['rt index'].append(np.where(rt_index > 0)[0])
                #print(ret['rt index'])
            else:
                ret['score chrom'].append(score_vec * 0)
                ret['bit score'].append(score_vec * 0)

            #ret['chrom'][-1] = ret['chrom'][-1] * ret['score chrom'][-1]
            ret['chrom'][-1] = ret['chrom'][-1] * ret['bit score'][-1]

            #ret['MSscore'].append(np.max(ret['score chrom'][-1]))
            ret['MSscore'].append(np.max(ret['score chrom'][-1]))
            matrix = matrix * oligo.negative

        total_matrix_area = np.sum(lcms_data.matrix)
        total_peaks_area = np.sum(ret['chrom'])

        chromList = []
        for chrom, oligo, MSscore, rt in zip(ret['chrom'], self.__oligos, ret['MSscore'], ret['rt index']):
            d = {}
            d['peak area'] = np.sum(chrom)
            if d['peak area'] > 0:
                d['peak area %'] = d['peak area'] * 100 / total_peaks_area
                if len(rt) > 0:
                    d['rt min'] = rt[0]
                    d['rt max'] = rt[-1]
                else:
                    d['rt min'] = 0
                    d['rt max'] = 0
                d['peak area total %'] = d['peak area'] * 100 / total_matrix_area
                d['oligo name'] = oligo.name
                d['oligo type'] = oligo.type
                d['score'] = MSscore
                d['oligo mass'] = oligo.mass
                d['oligo seq'] = oligo.seq
                ret['data'].append(d)
                chromList.append(chrom)
                #print(d['oligo name'], d['oligo type'], d['score'])

        ret['data'] = pd.DataFrame(ret['data'])
        ret['chrom'] = np.array(chromList)

        return ret

    def group_data(self, data):

        df = data.groupby('oligo type').agg({'peak area': 'sum', 'peak area %': 'sum', 'rt min': 'max', 'rt max': 'max',
                                              'peak area total %': 'sum', 'score': 'max',
                                              'oligo mass': 'min', 'oligo seq': 'first'})
        df.reset_index(inplace=True)

        return df

    def group2vec(self, data):
        out_d = {}
        #print(data[data['oligo name'].isin(['main', 'main dimer', 'main trimer', 'main tetramer'])])

        out_d['main'] = data[data['oligo name'].isin(['main', 'main+Na adducts',
                                                              'main dimer',
                                                              'main trimer',
                                                              'main tetramer'])]['peak area %'].sum()
        out_d['like main'] = data[data['oligo name'].str.contains('adducts', regex=False)&
                                  ~data['oligo name'].str.contains('Na adducts', regex=False)&
                                  (data['oligo type'] == 'main')]['peak area %'].sum()
        #print(data['oligo name'].str.contains('adducts', regex=False))
        out_d['aPurin'] = 0
        out_d['Del'] = data[data['oligo type'] == 'del']['peak area %'].sum()
        #print(out_d)
        nsum = 0
        for i in range(5):
            out_d[f'3 end n-{i + 1}'] = data[data['oligo type'].isin([f'3n-{i + 1}',
                                                                f'3n-{i + 1} adducts'])]['peak area %'].sum()
            nsum += out_d[f'3 end n-{i + 1}']
            out_d[f'5 end n-{i + 1}'] = data[data['oligo type'].isin([f'5n-{i + 1}',
                                                                f'5n-{i + 1} adducts'])]['peak area %'].sum()
            nsum += out_d[f'5 end n-{i + 1}']

        out_d['n-x'] = data[data['oligo type'].str.contains('n-', regex=False)]['peak area %'].sum()
        out_d['n-x'] = out_d['n-x'] - nsum
        #out_d['unknown'] = self.gTab[self.gTab['type'] == 'unknown']['purity%'].sum()
        return [out_d]



class oligoSynth1():
    def __init__(self, sequence):
        self.sequence = sequence
        self.oligos = []

    def run(self):

        oligo = mmo.oligoNASequence(self.sequence)
        print(oligo.getSeqTabDF())

        for i in range(1, oligo.getSeqLength()):

            prefix = oligo.getPrefix(i)
            suffix = oligo.getSuffix(i)
            if i > 1:
                deletion = oligo.getDeletion(i)
            else:
                deletion = ''
            print(prefix.sequence, suffix.sequence, deletion)


def test1():
    synt1 = oligoSynth1('{6FAM}ACGTACGT{BHQ1}')
    synt1.run()

def test2():
    o1 = oligoLeaf(seq = 'TTTTTTTTTTTTTTTTTT')
    o1.get_mz_forms(min_mz=600, max_mz=3200)
    o1.set_vector(3200)
    print(o1.vector)
    print(np.sum(o1.vector))
    print(o1.mass)
    print(pd.DataFrame(o1.get_mz_forms(min_mz=600, max_mz=3200)))

def test3():
    import msvis

    lcms = lcmsData('/home/alex/Documents/LCMS/oligos/synt/050822/DMTdT18_c1_52_f11.mzdata.xml')
    #lcms = lcmsData('/home/alex/Documents/LCMS/oligos/synt/180822/13.mzdata.xml')
    lcms.substruct_bkg(treshold=500)
    lcms.ms2matrix_1()
    data = lcms.data
    #print(np.round(data[:, 0]))
    #viewer = msvis.bokeh_mass_map(data[:, 0], data[:, 1], data[:, 2], rt_position=0, title='LCMS 2D map',
    #                        corner_points={'rt': [0, 1500], 'mz': [100, 2000]}, colorMap='cividis')
    #viewer.draw_map(is_show=True)

    o1 = oligoLeaf(seq='TTTTTTTTTTTTTTTTTT')
    #o1 = oligoLeaf(seq='ttcgggctttgttagcagccggatcctcgagctatttcttttgcttttctaacatttgca')
    o1.get_mz_forms(min_mz=120, max_mz=lcms.matrix[:, 0].shape[0])
    o1.set_vector(lcms.matrix[:, 0].shape[0])

    o2 = oligoLeaf(seq='TTTTTTTTTTTTTTTTT')
    o2.get_mz_forms(min_mz=120, max_mz=lcms.matrix[:, 0].shape[0])
    o2.set_vector(lcms.matrix[:, 0].shape[0])

    o3 = oligoLeaf(seq='{Na}TTTTTTTTTTTTTTTTTT')
    o3.get_mz_forms(min_mz=120, max_mz=lcms.matrix[:, 0].shape[0])
    o3.set_vector(lcms.matrix[:, 0].shape[0])

    print(lcms.matrix.shape)
    print(o1.vector.shape)
    print(pd.DataFrame(o1.forms))
    print(len(o1.forms['charge']))
    print(len(o1.forms['mz']))
    print(len(o1.forms['mz int']))

    #print(lcms.matrix.T)
    init_sum = np.sum(lcms.matrix)
    matrix = lcms.matrix.T
    Y1 = matrix @ o1.vector
    matrix = matrix * o1.negative
    Y11 = matrix @ o1.vector
    Y2 = matrix @ o2.vector
    matrix = matrix * o2.negative
    Y3 = matrix @ o3.vector
    matrix = matrix * o3.negative

    print(Y1.shape)
    print(np.sum(Y1))
    print(np.sum(lcms.matrix))
    print(np.sum(Y1) * 100 / init_sum)
    print(np.sum(Y11) * 100 / init_sum)
    print(np.sum(Y2) * 100 / init_sum)
    print(np.sum(Y3) * 100 / init_sum)

    import matplotlib.pyplot as plt

    plt.plot(range(Y1.shape[0]), Y1, '-')
    plt.plot(range(Y2.shape[0]), Y2, '-')
    plt.plot(range(Y3.shape[0]), Y3, '-')
    plt.show()

def test4():
    oligo = mmo.oligoNASequence('{DMT}TTTGGGTTCCGTCCTAAACGG')
    oligo2 = mmo.oligoNASequence('TTTGGGTTCCGTCCTAAACGG')
    print(oligo.getAvgMass() - oligo2.getAvgMass())
    print(oligo2.getAvgMass())
    print(oligo.getAvgMass())

def test5():
    #lcms = lcmsData('/home/alex/Documents/LCMS/oligos/synt/050822/DMTdT18_c1_52_f11.mzdata.xml',
    #                init_data=None)
    lcms = lcmsData('/home/alex/Documents/LCMS/oligos/synt/010922/11.mzdata.xml',
                    init_data=None)
    lcms.substruct_bkg(treshold=500, max_rt=1440, min_rt=240, int_treshold=600, min_mz=400)
    lcms.ms2matrix_1()

    #oligos = oligoTree(sequence='TTTTTTTTTTTTTTTTTT', min_mz=120, max_mz=lcms.mz_max,
    oligos = oligoTree(sequence='ATGCAGCTAATACGACTCACTATAGGACATTTGCTTCTAGCTTCTGAAACACGGT', min_mz=400, max_mz=lcms.mz_max,
                       vector_shape=lcms.vector_shape())

    results = oligos.mul(lcms)

    print(results)

    #import matplotlib.pyplot as plt

    #for chrom in results['chrom']:
    #    plt.plot(range(chrom.shape[0]), chrom, '-')
    #plt.show()

    import msvis
    plotter = msvis.charts1D_bokeh(results['score chrom'])
    plotter.draw(is_show=True)

if __name__=='__main__':
    test5()
