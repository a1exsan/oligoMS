import xml.etree.ElementTree as ET
import base64, struct
from tqdm import tqdm
import numpy as np

class spec_data():

    def __init__(self):
        self.id = None
        self.Polarity = None
        self.RT = None
        self.mz_dict = None
        self.intens_dict = None
        self.spectrumInstrument = None
        self.is_mz_dict = False

class mzdata():

    def __init__(self, fn):
        self.data = []
        with open(fn) as f:
            self.root = ET.parse(f).getroot()
            self._parse()

    def _parse(self):
        self.max_mz = 0
        for i in self.root.iter():
            if i.tag == 'spectrum':
                spectrum = spec_data()
                spectrum.id = i.attrib['id']
                #print(i.tag, i.attrib)
            if i.tag == 'spectrumInstrument':
                #print(i.tag, i.attrib)
                spectrum.spectrumInstrument = i.attrib
            if i.tag == 'cvParam':
                #print(i.tag, i.attrib)
                if i.attrib['accession'] == 'PSI:1000037':
                    spectrum.Polarity = i.attrib['value']
                if i.attrib['accession'] == 'PSI:1000038':
                    rt = float(i.attrib['value'].replace(',', '.'))
                    if i.attrib['name'] == 'TimeInMinutes':
                        spectrum.RT = {'value': rt * 60, 'units': 'sec'}
                    else:
                        spectrum.RT = {'value': rt, 'units': i.attrib['name']}
            if i.tag == 'data':
                #print(i.tag, i.attrib, i.text)
                if not spectrum.is_mz_dict:
                    spectrum.mz_dict = i.attrib
                    #spectrum.mz_dict['mz_text'] = i.text
                    spectrum.is_mz_dict = True

                    #print(str.encode(i.text))
                    raw_data = base64.decodebytes(str.encode(i.text))
                    out = struct.unpack('<%sd' % (len(raw_data) // 8), raw_data)
                    spectrum.mz_dict['mz_list'] = list(out)

                    mmax = max(spectrum.mz_dict['mz_list'])
                    if self.max_mz < mmax:
                        self.max_mz = mmax
                    #print(out)

                else:
                    spectrum.intens_dict = i.attrib
                    #spectrum.intens_dict['intens_text'] = i.text

                    raw_data = base64.decodebytes(str.encode(i.text))
                    out = struct.unpack('<%sf' % (len(raw_data) // 4), raw_data)
                    spectrum.intens_dict['intens_list'] = list(out)

                    self.data.append(spectrum)

        #for spec in self.data:
        #    print(spec.intens_dict)

    def mzdata2tab(self, int_treshold=5000, max_mz=3200, rt_left=100):

        vec = [0 for i in range(int(round(max_mz, 0)))]

        data = []
        for s in tqdm(self.data):
            rt = s.RT['value']
            if rt >= rt_left:
                for mz, intens in zip(s.mz_dict['mz_list'], s.intens_dict['intens_list']):
                    if intens >= int_treshold:
                        v = [rt, mz, intens]
                        data.append(v)
                        vec[int(round(mz, 0))] += 1

        return np.array(data), vec

    def mzdata2tab_all(self, int_treshold=5000):

        vec = [0 for i in range(int(round(self.max_mz, 0)) + 10)]

        data = []
        for s in tqdm(self.data):
            rt = s.RT['value']
            for mz, intens in zip(s.mz_dict['mz_list'], s.intens_dict['intens_list']):
                if intens >= int_treshold:
                    v = [rt, mz, intens]
                    data.append(v)
                    vec[int(round(mz, 0))] += 1

        return np.array(data), vec

def test1():
    spec = mzdata('/home/alex/Documents/LCMS/oligos/synt/300522/NP_c1_f3.mzdata')
    data, vec = spec.mzdata2tab()
    print(data)

def test2():

    import pandas as pd
    import matplotlib.pyplot as plt

    spec = mzdata('/home/alex/Documents/LCMS/oligos/synt/220622/dT18_c2_4.mzdata.xml')
    data, vec = spec.mzdata2tab()
    df = pd.DataFrame(data=data, columns=['rt', 'mz', 'intens'])
    df = df[(df['rt']>=1000)&(df['rt']<=1080)]
    df = df[(df['mz']>=1700)&(df['mz']<=1750)]
    print(df)

    plt.scatter(df['rt'], df['mz'], s=1)
    plt.show()

    df.to_csv('/home/alex/Documents/LCMS/test_data.csv', index=False)



if __name__ == '__main__':
    test2()