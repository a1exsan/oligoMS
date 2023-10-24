import pandas as pd
import numpy as np
import streamlit as st

import oligoMSExplainer as MSE
import oligoMass.molmassOligo as mmo
import mzdatapy as mzdata

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

import msvis

st.set_page_config(layout="wide")

@st.cache#(allow_output_mutation=True)
def upload_lcms_data(file_name):
    spec = mzdata.mzdata(file_name)
    return spec.mzdata2tab_all(int_treshold=0)

is_positive_mode = st.sidebar.checkbox('Positive mode')

sequence = st.sidebar.text_area('Enter sequence', '')

if sequence != '':
    oligos1 = mmo.oligoNASequence(sequence)
    st.sidebar.write(f'Molecular mass: {oligos1.getAvgMass():.2f} Da')
    st.sidebar.write(f'Extinction: {oligos1.getExtinction()} oe/mol')

uploaded_file = st.sidebar.file_uploader("Choose a file (*.mzML)")

col1, col2 = st.columns([3, 1])

with col2:
    clear_bkg = st.checkbox('clear background')

    rt_interval = st.select_slider('Retention time interval', options=range(0, 3010, 10), value=(50, 1500))

    min_mz = st.select_slider('min MZ', options=range(0, 2000, 10), value=200)

    if is_positive_mode:
        bkg_treshold = st.select_slider('select background treshold', options=range(100, 6000, 100), value=3500)
    else:
        bkg_treshold = st.select_slider('select background treshold', options=range(100, 6000, 100), value=500)

    if is_positive_mode:
        low_intens_treshold = st.select_slider('low intensity treshold', options=range(1000, 100000, 1000), value=7000)
        st.write('low intensity treshold', low_intens_treshold)
    else:
        low_intens_treshold = st.select_slider('low intensity treshold', options=range(0, 10000, 100), value=600)
        st.write('low intensity treshold', low_intens_treshold)

    score_treshold = st.select_slider('score treshold', options=range(0, 100, 1), value=10)
    score_treshold = score_treshold / 100
    st.write('score treshold', score_treshold)

    is_identify = st.checkbox('Identifying')

    colorMap = st.radio('Select Color Map',
                        ('monochrome', 'viridis', 'magma', 'inferno', 'cividis'))

    integ_start = st.text_area('Integrate TIC from:', 0, max_chars=10)
    integ_end = st.text_area('Integrate TIC to:', 0, max_chars=10)

with col1:
    if uploaded_file is not None:# and sequence != '':
        with open(f'data/temp/{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.getvalue())


        init_data, bkg = upload_lcms_data(f'data/temp/{uploaded_file.name}')
        lcms_data = MSE.lcmsData(init_data=init_data, bkg=bkg)
        data = lcms_data.data

        rt_pos = 0
        rt_left = rt_interval[0]
        rt_max = rt_interval[1]

        if clear_bkg:
            lcms_data.substruct_bkg(treshold=bkg_treshold, int_treshold=low_intens_treshold,
                                    max_rt=rt_max, min_rt=rt_left, min_mz=min_mz)
            lcms_data.ms2matrix_1()
            data = lcms_data.data

        lcms_data.ms2matrix_1()
        matrix = lcms_data.matrix.T
        TIC = [matrix @ np.ones(matrix.shape[1])]

        viewer = msvis.bokeh_mass_map(data[:, 0], data[:, 1], data[:, 2], rt_position=rt_pos, title='LCMS 2D map',
                                  corner_points={'rt': [rt_left, rt_max], 'mz': [100, 2000]}, colorMap=colorMap)
        viewer.draw_map(is_show=False)
        st.bokeh_chart(viewer.plot, use_container_width=True)

        tic_view = msvis.charts1D_bokeh(TIC, x_label='Retention time, sec',
                                        y_label='Intensity', title='TIC')
        tic_view.draw(is_show=False)
        st.bokeh_chart(tic_view.plot, use_container_width=True)

        purity = MSE.chromInteg(TIC[0], baseconst=100)
        st.write(f'set purity: {purity.integrate(int(integ_start), int(integ_end)):.2f}%')

        if is_identify and sequence != '':
            oligos = MSE.oligoTree(sequence=sequence, min_mz=min_mz, max_mz=lcms_data.mz_max,
                               vector_shape=lcms_data.vector_shape(), score_treshold=score_treshold)

            id_results = oligos.mul(lcms_data)
            grouped = oligos.group_data(id_results['data'])

            main_rt_min = grouped[grouped['oligo type'] == 'main']['rt min'].max()
            main_rt_max = grouped[grouped['oligo type'] == 'main']['rt max'].max()

            purity2 = MSE.chromInteg(TIC[0], baseconst=100)
            st.write(f'auto purity: {purity2.integrate(int(main_rt_min), int(main_rt_max)):.2f}%')

            #bscore_view = msvis.charts1D_bokeh(id_results['TIC'], x_label='Retention time, sec',
            #                                   y_label='Intensity', title='TIC')
            #bscore_view.draw(is_show=False)
            #st.bokeh_chart(bscore_view.plot, use_container_width=True)

            score_view = msvis.charts1D_bokeh(id_results['score chrom'], x_label='Retention time, sec',
                                              y_label='score', title='Score map')
            score_view.draw(is_show=False)
            st.bokeh_chart(score_view.plot, use_container_width=True)

            #bscore_view = msvis.charts1D_bokeh(id_results['bit score'], x_label='Retention time, sec',
            #                                  y_label='bscore', title='bitScore map')
            #bscore_view.draw(is_show=False)
            #st.bokeh_chart(bscore_view.plot, use_container_width=True)

            chrom_view = msvis.charts1D_bokeh(id_results['chrom'], x_label='Retention time, sec',
                                              y_label='Total intensity', title='Chromatograms')
            chrom_view.draw(is_show=False)
            st.bokeh_chart(chrom_view.plot, use_container_width=True)

            st.write('Table of identified Peaks:')
            st.write(id_results['data'])

            st.download_button('Download identified Peak as csv', id_results['data'].to_csv(sep='\t',
                                                                       index=False).encode('utf-8'),
                               f"{uploaded_file.name}_vector_results.csv",
                               "text/csv",
                               key='download-csv'
                               )

            st.write('Table of grouped Peaks:')
            st.write(grouped)

            st.download_button('Download Table as csv', grouped.to_csv(sep='\t',
                                                                               index=False).encode('utf-8'),
                               f"{uploaded_file.name}_vector_results.csv",
                               "text/csv",
                               key='download-csv1'
                               )


            vec_df = pd.DataFrame(oligos.group2vec(id_results['data']))
            st.write(vec_df)

            copy_button_1 = Button(label="Copy to clipboard w/o headers")
            copy_button_1.js_on_event("button_click", CustomJS(args=dict(df=vec_df.to_csv(sep='\t',
                                                                                              index=False,
                                                                                              header=False)), code="""
                            navigator.clipboard.writeText(df);
                            """))

            no_event_1 = streamlit_bokeh_events(
                copy_button_1,
                events="GET_TEXT",
                key="get_text_1",
                refresh_on_update=True,
                override_height=75,
                debounce_time=0)

            st.download_button('Download vector results as csv', vec_df.to_csv(sep='\t',
                                                                                     index=False).encode('utf-8'),
                               f"{uploaded_file.name}_vector_results.csv",
                               "text/csv",
                               key='download-csv2'
                               )
