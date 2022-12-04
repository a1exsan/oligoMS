import pandas as pd
import streamlit as st

import oligoMSExplainer as MSE # для рассчета площади пика
import mzdatapy as mzdata  # для рассчета площади пика
import numpy as np # для рассчета площади пика

import oligoMass.molmassOligo as mmo
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

import oligoMS as lcms
import msvis

st.set_page_config(layout="wide")

@st.cache
def upload_mzML(name, sequence):
    return lcms.lcms_peptide(name, sequence)

@st.cache#(allow_output_mutation=True)
def upload_lcms_data(file_name):
    spec = mzdata.mzdata(file_name)
    return spec.mzdata2tab_all(int_treshold=0)

@st.cache
def upload_mzML_data(name, rt_left=100):
    return lcms.openMSdata(name, int_treshold=1000, max_mz=3200, rt_left=rt_left)
    #lcms.open_mzml(name, int_treshold=5000, max_mz=3200, rt_left=rt_left)

@st.cache
def deconvolution(data, is_positive):
    deconv = lcms.oligosDeconvolution(data[:, 0], data[:, 1], data[:, 2], is_positive=is_positive)
    return deconv.deconvolute(), deconv


is_positive_mode = st.sidebar.checkbox('Positive mode')
#is_positive_mode = False
sequence = st.sidebar.text_area('Enter sequence', '')

if sequence != '':
    oligos1 = mmo.oligoNASequence(sequence)
    st.sidebar.write(f'Molecular mass: {oligos1.getAvgMass():.2f} Da')
    st.sidebar.write(f'Extinction: {oligos1.getExtinction()} oe/mol')

uploaded_file = st.sidebar.file_uploader("Choose a file (*.mzML)")

col1, col2 = st.columns([3, 1])

with col2:
    clear_bkg = st.checkbox('clear background')

    polish_bkg = st.checkbox('polish background')

    rt_interval = st.select_slider('Retention time interval', options=range(0, 2010, 10), value=(100, 1500))

    if is_positive_mode:
        bkg_treshold = st.select_slider('select background treshold', options=range(100, 6000, 100), value=3500)
    else:
        bkg_treshold = st.select_slider('select background treshold', options=range(100, 6000, 100), value=500)

    if is_positive_mode:
        neighbor_treshold = st.select_slider('select neighbor treshold', options=range(10, 100, 5), value=35)
    else:
        neighbor_treshold = st.select_slider('select neighbor treshold', options=range(10, 100, 5), value=60)

    if is_positive_mode:
        low_intens_treshold = st.select_slider('low intensity treshold', options=range(100, 10000, 100), value=7000)
        st.write('low intensity treshold', low_intens_treshold)
    else:
        low_intens_treshold = st.select_slider('low intensity treshold', options=range(100, 10000, 100), value=6000)
        st.write('low intensity treshold', low_intens_treshold)

    if is_positive_mode:
        bkg_polish_count = st.select_slider('background polish repeats', options=range(1, 7, 1), value=1)
    else:
        bkg_polish_count = st.select_slider('background polish repeats', options=range(1, 7, 1), value=3)

    is_deconv = st.checkbox('Deconvolution')

    is_identify = st.checkbox('Identifying')

    is_droped = st.checkbox('drop failed points')

    mass_treshold = st.select_slider('select background treshold', options=range(200, 1010, 10), value=500)

    is_drop_unk = st.checkbox('Drop unknown')

    score_treshold = st.select_slider('score treshold', options=range(0, 100, 1), value=10)   # для рассчета площади пика
    score_treshold = score_treshold / 100                                                     # для рассчета площади пика
    st.write('score treshold', score_treshold)                                                # для рассчета площади пика

    integ_start = st.text_area('Integrate TIC from:', 0, max_chars=10)  # для рассчета площади пика
    integ_end = st.text_area('Integrate TIC to:', 0, max_chars=10)      # для рассчета площади пика

    # viridis, magma, inferno, cividis
    colorMap = st.radio('Select Color Map',
                        ('monochrome', 'viridis', 'magma', 'inferno', 'cividis'))

with col1:
    if uploaded_file is not None:# and sequence != '':
        with open(f'data/temp/{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.getvalue())

        rt_left = rt_interval[0]
        data, bkg = upload_mzML_data(f'data/temp/{uploaded_file.name}', rt_left=rt_left)

        init_data, bkg1 = upload_lcms_data(f'data/temp/{uploaded_file.name}') # для рассчета площади пика
        lcms_data = MSE.lcmsData(init_data=init_data, bkg=bkg1)               # для рассчета площади пика
        data1 = lcms_data.data                                                # для рассчета площади пика

        if clear_bkg:
            data = lcms.substract_bkg(data, bkg, treshold=bkg_treshold)

            lcms_data.substruct_bkg(treshold=bkg_treshold, int_treshold=low_intens_treshold,    # для рассчета площади пика
                                    max_rt=rt_interval[1], min_rt=rt_left, min_mz=200)
            lcms_data.ms2matrix_1()                                                             # для рассчета площади пика
            data1 = lcms_data.data                                                             # для рассчета площади пика

        lcms_data.ms2matrix_1()                      # для рассчета площади пика
        matrix = lcms_data.matrix.T                  # для рассчета площади пика
        TIC = [matrix @ np.ones(matrix.shape[1])]    # для рассчета площади пика

        if polish_bkg:
            for i in range(bkg_polish_count):
                map = lcms.get_intensity_map(data, low_treshold=low_intens_treshold, param=4)
                data = lcms.find_inner_points(data, map, neighbor_treshold=neighbor_treshold, param=4)

        if is_deconv:
            deconv_data, deconv_obj = deconvolution(data, is_positive=is_positive_mode)
            deconv_data = deconv_obj.rt_filtration(deconv_data, rt_interval[0], rt_interval[1])

        if is_identify and sequence != '':

            if is_positive_mode:
                explainer = lcms.peptideMassExplainer(sequence, deconv_data)
                explainer.explain_2(mass_treshold=1)
                explainer.group_by_type_2()
            else:
                explainer = lcms.oligoMassExplainer2(sequence, deconv_data)
                explainer.explain_2(mass_treshold=2)
                explainer.group_by_type_2()

            if is_droped:
                if not is_positive_mode:
                    explainer.drop_artifacts(mass_thold=mass_treshold)
                    explainer.explain_2(mass_treshold=2)
                    explainer.group_by_type_2()


        if is_drop_unk and is_identify and sequence != '':
            deconv_data = explainer.drop_unknown(explainer.mass_tab)

            if is_positive_mode:
                explainer = lcms.peptideMassExplainer(sequence, deconv_data)
                explainer.explain_2(mass_treshold=1)
                explainer.group_by_type_2()
            else:
                explainer = lcms.oligoMassExplainer2(sequence, deconv_data)
                explainer.explain_2(mass_treshold=2)
                explainer.group_by_type_2()


        rt_max = int(round(data[:, 0].max(), 0))

        rt_pos = st.select_slider('select spectra', options=range(0, rt_max, 1))

        viewer = msvis.bokeh_mass_map(data[:, 0], data[:, 1], data[:, 2], rt_position=rt_pos, title='LCMS 2D map',
                                      corner_points={'rt': [rt_left, rt_max], 'mz': [100, 2000]}, colorMap=colorMap)
        viewer.draw_map(is_show=False)
        st.bokeh_chart(viewer.plot, use_container_width=True)

        tic_view = msvis.charts1D_bokeh(TIC, x_label='Retention time, sec',               # для рассчета площади пика
                                        y_label='Intensity', title='TIC')                 # для рассчета площади пика
        tic_view.draw(is_show=False)
        st.bokeh_chart(tic_view.plot, use_container_width=True)                           # для рассчета площади пика

        purity = MSE.chromInteg(TIC[0], baseconst=100)                                      # для рассчета площади пика
        st.write(f'set purity: {purity.integrate(int(integ_start), int(integ_end)):.2f}%')  # для рассчета площади пика

        if sequence != '':
            oligos = MSE.oligoTree(sequence=sequence, min_mz=200, max_mz=lcms_data.mz_max,      # для рассчета площади пика
                                   vector_shape=lcms_data.vector_shape(), score_treshold=score_treshold)  # для рассчета площади пика

            id_results = oligos.mul(lcms_data)                                                    # для рассчета площади пика
            grouped = oligos.group_data(id_results['data'])                                       # для рассчета площади пика

            main_rt_min = grouped[grouped['oligo type'] == 'main']['rt min'].max()                # для рассчета площади пика
            main_rt_max = grouped[grouped['oligo type'] == 'main']['rt max'].max()                # для рассчета площади пика

            purity2 = MSE.chromInteg(TIC[0], baseconst=100)                                         # для рассчета площади пика
            st.write(f'auto purity: {purity2.integrate(int(main_rt_min), int(main_rt_max)):.2f}%')  # для рассчета площади пика


        if is_deconv:
            spec_viewer = msvis.bokeh_ms_spectra(deconv_data['rt'], deconv_data['mono_mass'],
                                                 deconv_data['intens'], rt_position=rt_pos)
            spec_viewer.draw_map(is_show=False)
            st.bokeh_chart(spec_viewer.plot, use_container_width=True)
        else:
            spec_viewer = msvis.bokeh_ms_spectra(data[:, 0], data[:, 1],
                                                 data[:, 2], rt_position=rt_pos)
            spec_viewer.draw_map(is_show=False)
            st.bokeh_chart(spec_viewer.plot, use_container_width=True)

        if is_deconv and not is_identify:
            mass_viewer = msvis.bokeh_mass_map(deconv_data['rt'],
                                                deconv_data['mono_mass'],
                                                deconv_data['intens'], rt_position=-1, title='Deconvolution 2D map',
                                                corner_points={'rt': [rt_left, rt_max], 'mz': [100, 3000]},
                                               colorMap=colorMap)
            mass_viewer.yaxis_label = 'Mass, Da'
            mass_viewer.draw_map(is_show=False)
            st.bokeh_chart(mass_viewer.plot, use_container_width=True)

        elif is_deconv and is_identify:
            mass_viewer = msvis.bokeh_mass_map(explainer.mass_tab['rt'],
                                               explainer.mass_tab['mono_mass'],
                                               explainer.mass_tab['intens'], rt_position=-1, title='Deconvolution 2D map',
                                               corner_points={'rt': [rt_left, rt_max], 'mz': [100, 3000]},
                                               colorMap=colorMap)
            mass_viewer.yaxis_label = 'Mass, Da'
            mass_viewer.draw_map(is_show=False)
            st.bokeh_chart(mass_viewer.plot, use_container_width=True)


        if is_identify:
            st.write('Table of identified Peaks:')
            st.write(explainer.mass_tab)

            st.write('Table of identified Groups:')
            #st.write(explainer.gTab)
            st.dataframe(explainer.gTab)

            st.download_button('Download results as csv', explainer.gTab.to_csv().encode('utf-8'),
                               f"{uploaded_file.name}_results.csv",
                               "text/csv",
                               key='download-csv'
                               )


            # Adding copy to clipboard


            lcms_pipeline = lcms.oligoPipeline(explainer.gTab, explainer.seq)
            lcms_pipeline.pipeline()
            lcms_results = pd.DataFrame([lcms_pipeline.group_inpurit()])

            lcms_results['clear_bkg'] = clear_bkg
            lcms_results['polish_bkg'] = polish_bkg
            lcms_results['rt_interval'] = str(rt_interval)
            lcms_results['bkg_treshold'] = str(bkg_treshold)
            lcms_results['is_positive_mode'] = is_positive_mode
            lcms_results['neighbor_treshold'] = neighbor_treshold
            lcms_results['low_intens_treshold'] = low_intens_treshold
            lcms_results['bkg_polish_count'] = bkg_polish_count
            lcms_results['is_deconv'] = is_deconv
            lcms_results['is_identify'] = is_identify
            lcms_results['is_droped'] = is_droped
            lcms_results['mass_treshold'] = mass_treshold
            lcms_results['is_drop_unk'] = is_drop_unk

            st.dataframe(lcms_results)

            copy_button = Button(label="Copy to clipboard")
            copy_button.js_on_event("button_click", CustomJS(args=dict(df=lcms_results.to_csv(sep='\t',
                                                                                              index=False)), code="""
                navigator.clipboard.writeText(df);
                """))

            no_event = streamlit_bokeh_events(
                copy_button,
                events="GET_TEXT",
                key="get_text",
                refresh_on_update=True,
                override_height=75,
                debounce_time=0)

            #
            copy_button_1 = Button(label="Copy to clipboard w/o headers")
            copy_button_1.js_on_event("button_click", CustomJS(args=dict(df=lcms_results.to_csv(sep='\t',
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

