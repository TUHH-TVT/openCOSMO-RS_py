import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import opencosmorspy.segtp_collection as stpc
from opencosmorspy.parameterization import Parameterization
from opencosmorspy.molecules import Molecule

def _generate_extended_sigma_profile(filepath, qc_program, par=None):

    mol = Molecule([filepath], qc_program)
    # ORCA

    if par is None:
        if qc_program == 'orca':
            par = Parameterization('default_orca')
        elif qc_program == 'turbomole':
            par = Parameterization('default_turbomole')

    mol.convert_properties(par.r_av, par.mf_r_av_corr)

    segtp_col = stpc.SegtpCollection(par)
    segtp_col.cluster_cosmo_struct(mol.cosmo_struct_lst[0], par.sigma_grid,
                                   par.sigma_orth_grid)
    df_esp = pd.DataFrame(segtp_col.get_segtps_as_array_dct())

    df_esp['area'] = 0.
    segtp_idxs = list(mol.cosmo_struct_lst[0].segtp_area_dct.keys())
    segtp_areas = list(mol.cosmo_struct_lst[0].segtp_area_dct.values())
    df_esp.loc[segtp_idxs, 'area'] = list(segtp_areas)



    return df_esp

def _generate_sigma_profile(filepath, qc_program, par=None):
    
    if par is None:
        if qc_program == 'orca':
            par = Parameterization('default_orca')
        elif qc_program == 'turbomole':
            par = Parameterization('default_turbomole')
    
    df_esp = _generate_extended_sigma_profile(filepath, qc_program, par)
    
    df_sp = pd.pivot_table(df_esp, values='area', index='sigma')
    df_sp.reset_index(inplace=True)

    for sigma in par.sigma_grid:
        if np.all(np.abs(sigma - df_sp['sigma']) > 1e-12):
            idx_new = df_sp.index[-1]+1
            df_sp.loc[idx_new, 'sigma'] = sigma
            df_sp.loc[idx_new, 'area'] = 0.
    
    df_sp.sort_values('sigma', inplace=True)
    
    
    return df_sp
    

def plot_sigmaprofiles(dir_plot, filepath_lst, qc_program, plot_name=None,
                       plot_label_dct=None, xlim=(-0.02, 0.02)):
    
    if plot_name is None:
        plot_name = 'sigma_profiles'
    
    if type(qc_program) == str:
        qc_program_lst = [qc_program for filepath in filepath_lst]
    else:
        qc_program_lst = [qc_program[filepath] for filepath in filepath_lst]
    
    plot_label_lst = []
    for filepath in filepath_lst:
        if plot_label_dct is not None and filepath in plot_label_dct:
            plot_label_lst.append(plot_label_dct[filepath])
        else:
            plot_label_lst.append(os.path.basename(filepath))

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for filepath, qcp, label in zip(filepath_lst, qc_program_lst,
                                    plot_label_lst):
        df_sp = _generate_sigma_profile(filepath, qcp)
        ax.plot(df_sp['sigma'], df_sp['area'], label=label)

    ax.set_xlim(*xlim)

    plt.show()



def plot_sigmaprofiles_plotly(dir_plot, filepath_lst, qc_program,
                              mode='static',
                              plot_name=None, plot_label_dct=None,
                              xlim=(-0.02, 0.02)):
    
    if plot_name is None:
        plot_name = 'sigma_profiles'
    
    if type(qc_program) == str:
        qc_program_lst = [qc_program for filepath in filepath_lst]
    else:
        qc_program_lst = [qc_program[filepath] for filepath in filepath_lst]
    
    plot_label_lst = []
    for filepath in filepath_lst:
        if plot_label_dct is not None and filepath in plot_label_dct:
            plot_label_lst.append(plot_label_dct[filepath])
        else:
            plot_label_lst.append(os.path.basename(filepath))


    fontsize = 20
    fontsize_tick = 20
    
    if mode == 'dynamic':
        xaxis_title = 'sigma'
        yaxis_title = 'p(sigma)'
    elif mode == 'static':
        xaxis_title = '$\large \sigma \; [e/ 10^{-10} m]$'
        yaxis_title = '$\large p(\sigma) \; [-]$'
    
    fig = go.Figure()
    
    
    y_max = 0
    for idx, (filepath, qcp, label) in enumerate(zip(
            filepath_lst, qc_program_lst, plot_label_lst)):
        
        if idx == 0:
            line={
                'dash': 'solid',
                'color': 'black'}  # blue '#316395' 
        elif idx == 1:
            line={
                'dash': 'longdash',
                'color': '#AF0000'}
        elif idx == 2:
            line={
                'dash': 'dashdot',
                'color': '#109618'}
        elif idx == 4:
            line={
                'dash': 'dot',
                'color': '#09118C'}
        elif idx == 5:
            line={
                'dash': 'dash',
                'color': '#7600b5'}

        elif idx == 5:
            line={
                'dash': 'longdashdot',
                'color': '#DEBC00'}
        elif idx == 6:
            line={
                'dash': 'dot',
                'color':  '#565656'}
        
        df_sp = _generate_sigma_profile(filepath, qcp)
        y_max = max(y_max, df_sp['area'].max())
        
        basename = os.path.basename(filepath)

        fig.add_trace(
        go.Scatter(x=df_sp['sigma'],
                   y=df_sp['area'],
                   mode='lines',
                   name=basename,
                   line=line)
                  )

    fig.update_layout(
        width=700,
        height=500,
        xaxis_title = xaxis_title,
        yaxis_title = yaxis_title,
        # margin=dict(
        #     l=50,
        #     r=50,
        #     b=50,
        #     t=50,
        #     pad=4),
        )
    fig.update_layout({
        'xaxis_range':[-0.02, 0.02],
        'yaxis_range':[0, y_max*1.2],
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'White',
        'font': {'size': fontsize,
                 'color': 'rgba(1, 1, 1, 1)'},
        'legend_title_text': '',
        'showlegend': False,
        'legend': {
            'yanchor':"top",
            'y': 0.98,
            'xanchor': 'right',
            'x': 0.98,
            'bgcolor' :'White',
            'bordercolor': 'Black',
            'borderwidth': 0.5
             }
        })
 
    fig.update_xaxes(tick0 = -0.02,
                     dtick = 0.005,
                     color = 'rgba(1, 1, 1, 1)',
                     gridcolor =  'rgba(0, 0, 0, 0.0)',
                     showgrid=False,
                     zeroline= False,
                     zerolinecolor='rgba(1, 1, 1, 1)',
                     zerolinewidth=0.5,
                     tickformat='.3f',
                     mirror=True,
                     linecolor='black',
                     ticks='outside',
                     showline=True)
    fig.update_yaxes(tick0 = 0,
                      # dtick = 1,
                      color = 'rgba(1, 1, 1, 1)',
                     gridcolor =  'rgba(0, 0, 0, 0.0)',
                     showgrid=False,
                     zeroline= True,
                     zerolinecolor='rgba(1, 1, 1, 1)',
                     zerolinewidth=0.5,
                     tickformat='.1f',
                     mirror=True,
                     linecolor='black',
                     ticks='outside',
                     showline=True)


    fig.show()
    if mode == 'static':
        fig.write_image(os.path.join(dir_plot, plot_name+'.png'))
        fig.write_image(os.path.join(dir_plot, plot_name+'.pdf'))
        fig.write_image(os.path.join(dir_plot, plot_name+'.svg'))
    if mode == 'dynamic':
        fig.write_html(os.path.join(dir_plot, plot_name+'.html'))
        
        
    


def plot_extended_sigmaprofiles_plotly(dir_plot, filepath, qc_program,
                                       mode='dynamic'):

    plot_name = 'extsp_'+os.path.splitext(os.path.basename(filepath))[0]
    
    df_esp = _generate_extended_sigma_profile(filepath, qc_program)

    df_esp['color'] = df_esp['elmnt_nr']
    df_esp.loc[df_esp['color'] == 106, 'color'] = 'H-C'
    df_esp.loc[df_esp['color'] == 108, 'color'] = 'H-O'
    df_esp.loc[df_esp['color'] == 107, 'color'] = 'H-N'
    df_esp.loc[df_esp['color'] == 6, 'color'] = 'C'
    df_esp.loc[df_esp['color'] == 8, 'color'] = 'O'
    df_esp.loc[df_esp['color'] == 7, 'color'] = 'N'
    df_esp.loc[df_esp['color'] == 17, 'color'] = 'Cl'

    # Make small areas visible, as they are clusters
    df_esp.loc[(df_esp['area'] != 0) & (df_esp['area'] < 0.06), 'area'] = 0.06

    # Pivot
    df_temp = pd.pivot_table(df_esp, values='sigma_orth', index='sigma',
                             aggfunc=np.sum)
    df_temp.reset_index(inplace=True)

    fontsize = 24
    fontsize_tick = 24

    # In static mode latex works very bad
    if mode == 'static':
        labels = {
            'sigma': r'$\Large\sigma [e/10^{-10} m]$',
            'sigma_orth': r'$\Large\sigma^{\perp} [e/10^{-10} m]$'
            }
    elif mode == 'dynamic':
        labels = {}

    fig = px.scatter(df_esp, x='sigma', y='sigma_orth', color='color',
                     labels=labels,
                     size='area',
                     hover_data=['sigma', 'sigma_orth', 'elmnt_nr'],
                     size_max=40,
                     opacity=0.7,
                     color_discrete_map={'H-C': '#316395',
                                         'H-O': '#AF0000',
                                         'H-N': '#7600b5',
                                         'C': '#109618',
                                         'O': '#565656',
                                         'N': '#DEBC00',
                                         'Cl': '#09118C'},
                     category_orders={'color': ['H-O', 'H-N', 'H-C',
                                                'Cl', 'C', 'N', 'O', 'Cl']},
                     width=800,
                     height=500
                     )
    fig.update_layout({
        'xaxis_range': [-0.025, 0.025],
        'yaxis_range': [-0.007, 0.007],
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'White',
        'font': {'size': fontsize,
                 'color': 'rgba(1, 1, 1, 1)'},
        'legend_title_text': '',
        'legend': {
            'yanchor': 'top',
            'y': 0.98,
            'xanchor': 'left',
            'x': 0.85,
            'bgcolor': 'White',
            'bordercolor': 'Black',
            'borderwidth': 0.5
             }
        })

    fig.update_xaxes(tick0=-0.020,
                     tickfont={'size': fontsize_tick},
                     dtick=0.01,
                     color='rgba(1, 1, 1, 1)',
                     gridcolor='rgba(0, 0, 0, 0.0)',
                     zeroline=False,
                     tickformat='.3f',
                     mirror=True,
                     linecolor='black',
                     ticks='outside',
                     showline=True)
    fig.update_yaxes(tick0=-0.006,
                     dtick=0.002,
                     tickfont={'size': fontsize_tick},
                     color='rgba(1, 1, 1, 1)',
                     gridcolor='rgba(0, 0, 0, 0.0)',
                     zeroline=False,
                     tickformat='.3f',
                     mirror=True,
                     linecolor='black',
                     ticks='outside',
                     showline=True)

    fig.show()
    if mode == 'static':
        fig.write_image(os.path.join(dir_plot, plot_name+'.png'))
        fig.write_image(os.path.join(dir_plot, plot_name+'.pdf'))
        fig.write_image(os.path.join(dir_plot, plot_name+'.svg'))
    if mode == 'dynamic':
        fig.write_html(os.path.join(dir_plot, plot_name+'.html'))


if __name__ == '__main__':
    pass