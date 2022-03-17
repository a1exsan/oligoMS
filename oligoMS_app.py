import streamlit as st

import oligoMS as lcms
import msvis

st.set_page_config(layout="wide")

@st.cache
def upload_mzML(name, sequence):
    return lcms.lcms_peptide(name, sequence)

@st.cache
def upload_mzML_data(name, rt_left=100):
    return lcms.open_mzml(name, int_treshold=5000, max_mz=3200, rt_left=rt_left)

@st.cache
def deconvolution(data, is_positive):
    deconv = lcms.oligosDeconvolution(data[:, 0], data[:, 1], data[:, 2], is_positive=is_positive)
    return deconv.deconvolute(), deconv


is_positive_mode = st.sidebar.checkbox('Positive mode')
#is_positive_mode = False
sequence = st.sidebar.text_area('Enter sequence', '')
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
        low_intens_treshold = st.select_slider('low intensity treshold', options=range(1000, 100000, 1000), value=7000)
        st.write('low intensity treshold', low_intens_treshold)
    else:
        low_intens_treshold = st.select_slider('low intensity treshold', options=range(1000, 100000, 1000), value=6000)
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

with col1:
    if uploaded_file is not None:# and sequence != '':
        with open(f'data/temp/{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.getvalue())

        rt_left = rt_interval[0]
        data, bkg = upload_mzML_data(f'data/temp/{uploaded_file.name}', rt_left=rt_left)

        if clear_bkg:
            data = lcms.substract_bkg(data, bkg, treshold=bkg_treshold)

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
                                      corner_points={'rt': [rt_left, rt_max], 'mz': [100, 2000]})
        viewer.draw_map(is_show=False)
        st.bokeh_chart(viewer.plot, use_container_width=True)

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
                                                corner_points={'rt': [rt_left, rt_max], 'mz': [100, 3000]})
            mass_viewer.ylabel = 'Mass, Da'
            mass_viewer.draw_map(is_show=False)
            st.bokeh_chart(mass_viewer.plot, use_container_width=True)

        elif is_deconv and is_identify:
            mass_viewer = msvis.bokeh_mass_map(explainer.mass_tab['rt'],
                                               explainer.mass_tab['mono_mass'],
                                               explainer.mass_tab['intens'], rt_position=-1, title='Deconvolution 2D map',
                                               corner_points={'rt': [rt_left, rt_max], 'mz': [100, 3000]})
            mass_viewer.ylabel = 'Mass, Da'
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

