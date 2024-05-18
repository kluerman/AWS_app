"""
Created on Sat Mar 27 12:45:25 2021

@author: kluerman
"""
import re
import glob
import os
import datetime

from flask import Flask, render_template, url_for, request, jsonify, Response, flash, session
import pandas as pd
from werkzeug.utils import redirect, secure_filename
import os
from itertools import islice, product

import string
import re
import numpy as np


import re
import glob
import os

from flask import Flask, render_template, url_for, request, jsonify, Response, flash, session
import pandas as pd
from werkzeug.utils import redirect, secure_filename
import os
from itertools import islice, product

import string
import re
import numpy as np


import pandas as pd
import numpy as np
from openpyxl import load_workbook



from bokeh.io import show
from bokeh.layouts import Column, row, gridplot, layout
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter, CustomJSTransform, MultiSelect, CustomJS
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show, from_networkx, save
from bokeh.models.graphs import from_networkx
from bokeh.models import LinearAxis, MultiChoice, Select, IndexFilter, Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, FactorRange, LabelSet, HoverTool
from bokeh.layouts import Column, layout, gridplot, column
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import cividis
from bokeh.transform import linear_cmap, dodge
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Slider,CheckboxGroup,DataTable,MultiSelect,TableColumn, Button, Panel, Tabs, TextInput
from bokeh.io import curdoc
from bokeh.embed import file_html, components, server_document
from bokeh.models import LinearColorMapper, ColorBar, Spinner
from bokeh.transform import transform
from bokeh.themes import built_in_themes
from bokeh.embed import server_document
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter, CustomJSTransform, MultiSelect, CustomJS, CustomJSFilter, RangeSlider, Panel, Tabs, LabelSet

import os, glob

import re

import warnings
warnings.filterwarnings('ignore')

def pick_left_numbers(old_n):
    if old_n is None:
        return None
    else:
        try: 
            digits = float(re.match("([0-9.-]*[0-9.])",old_n).groups()[0])
        except:
            digits = None
        return digits

def pick_right_characters(old_n):
    if old_n is None:
        return None
    else:
        try: 
            digits = old_n[len(re.match("([0-9.-]*[0-9.])",old_n).groups()[0]):]
        except:
            digits = None
        return digits
    
def export_csv(filename_sublist):

    labels = pd.read_excel('O:/SB_PKW_OM6xx/10_Tickets/1758-RDPRD-OM654-BRxx-Ladedruckregelabweichung_Anomalieerkennung_beanstandete_Fahrzeuge/03_Auswertungen/MRD1_Labeldefinition_Alex.xlsx')
    list_of_signals = list(labels.Label)

    datenstand = [get_datenstand_csv(filename) if ".csv" in filename
                    else get_datenstand_log(filename) for filename in filename_sublist]
    data = [read_measurement_csv(filename) if ".csv" in filename
          else read_measurement_log(filename) for filename in filename_sublist]
    list_of_signals = []
    
    [[list_of_signals.append(data[j][i].split(']')[-1].split(':')[0]) for i in range(len(data[j])) if len(data[j][i].split(']'))==2] for j in range(len(filename_sublist)) if ".log" in filename_sublist[j]]
    [list_of_signals.append(data[j].Signalname.unique().tolist()) for j in range(len(filename_sublist)) if ".csv" in filename_sublist[j]]
    list_of_signals = [item for sublist in list_of_signals for item in sublist]
    list_of_signals = list(set(list_of_signals))

    pruefstufen_dfs = [get_pruefstufen_and_ecus(filename) if ".log" in filename else np.nan for filename in filename_sublist]
    signal_df = pd.DataFrame({'Signalname':[], 'filename':[], 'Datenstand': [],'Pruefstufe': [], 'ECU':[], 'value':[], 'unit':[]})                                

    for signal in list_of_signals:
        # signal = list_of_signals[150]
        # ']' + signal + ':'
        # signal = 'Air_pCACDs_Read.Air_pCACDs'
        for i in range(len(filename_sublist)):
            if ".log" in filename_sublist[i]:
                signal_indices = [i for i,s in enumerate(data[i]) if ']' + signal + ':' in s]
                corresponding_pruefstufe = [pruefstufen_dfs[i][pruefstufen_dfs[i].Pruefstufe_start_index == np.array(pruefstufen_dfs[i].Pruefstufe_start_index)[np.array(pruefstufen_dfs[i].Pruefstufe_start_index) < signal_index].max()].Pruefstufe.to_list()[0] if np.sum([np.array(pruefstufen_dfs[i].Pruefstufe_start_index) < signal_index])>0 else np.nan for signal_index in signal_indices]
                pruefstufen_dict = dict(zip(corresponding_pruefstufe, signal_indices))
                corresponding_pruefstufe = list(pruefstufen_dict.keys())
                signal_indices = list(pruefstufen_dict.values())
                corresponding_ecu = [pruefstufen_dfs[i][pruefstufen_dfs[i].Pruefstufe_start_index == np.array(pruefstufen_dfs[i].Pruefstufe_start_index)[np.array(pruefstufen_dfs[i].Pruefstufe_start_index) < signal_index].max()].ECU.to_list()[0] if np.sum([np.array(pruefstufen_dfs[i].Pruefstufe_start_index) < signal_index])>0 else np.nan for signal_index in signal_indices]
                signal_values = [pick_left_numbers(data[i][signal_index].split(":")[-1].strip()) for signal_index in signal_indices]
                signal_units = [pick_right_characters(data[i][signal_index].split(":")[-1].strip()) for signal_index in signal_indices]
                signal_i_df = pd.DataFrame({'Signalname':signal, 'filename':filename_sublist[i], 'Datenstand': datenstand[i], 'Pruefstufe': corresponding_pruefstufe, 'ECU':corresponding_ecu, 'value':signal_values, 'unit':signal_units})                                
            else:
                signal_i_df = data[i][['Signalname','filename', 'Pruefstufe', 'ECU', 'value', 'unit']]
                signal_i_df['Datenstand'] = datenstand[i]
                signal_i_df = signal_i_df[['Signalname','filename','Datenstand', 'Pruefstufe', 'ECU', 'value', 'unit']]

            signal_df = signal_df.append(signal_i_df, ignore_index=True)
        signal_df['filename'] = signal_df['filename'].apply(lambda x: x.split('/')[-1])
    return signal_df

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_pruefstufen_and_ecus(filename):
    file_data = read_measurement_log(filename)
    pruefstufen_indices = [i for i,s in enumerate(file_data) if "Pruefstufe" in s]
    pruefstufen_names = [s for i,s in enumerate(file_data) if "Pruefstufe" in s]
    ecus_indices = [i for i,s in enumerate(file_data) if "ECU:" in s]
    ecus_names = [s for i,s in enumerate(file_data) if "ECU:" in s]

    corresponding_ecu_indices = [np.array(ecus_indices)[np.array(ecus_indices) < i].max() for i in pruefstufen_indices]
    corresponding_ecu_names = [ecus_names[ecus_indices.index(i)] for i in corresponding_ecu_indices]
    pruefstufen_df = pd.DataFrame({'Pruefstufe': pruefstufen_names, 'Pruefstufe_start_index':pruefstufen_indices, 'ECU':corresponding_ecu_names})
    return pruefstufen_df

def get_datenstand_csv(filename):
    file = open(filename, 'r', encoding="utf8", errors='ignore')
    logdata = file.read().splitlines()
    file.close()
    df = list(filter(None, logdata))

    datenstand = [s for s in df if "Datenstand" in s]

    if len(datenstand) == 0:
        datenstand = [s for s in df if "DT_0169_RDBIPathDataSet" in s]

    if len(datenstand) == 0:
        datenstand = "nicht verfügbar"
    else:

        datenstand = str(datenstand[0]).split(";")[1]

    return datenstand

def get_datenstand_log(filename):
    file = open(filename, 'r', encoding="utf8", errors='ignore')
    logdata = file.read().splitlines()
    file.close()
    logdata = list(filter(None, logdata))

    datenstand = [s for s in logdata if "[1000]Datenstand_Read.Datenstand" in s]
    if len(datenstand) == 0:
        datenstand = [s for s in logdata if "DT_0169_RDBIPathDataSet" in s]
        if len(datenstand) == 0:
            datenstand = [s for s in logdata if "Datensatzkennung" in s]

    datenstand = str(datenstand[0]).split("<")[0]

    if len(datenstand) == 0:
        datenstand = "nicht verfügbar"
    else:

        datenstand = str(datenstand).split(":")[1]
        datenstand = datenstand.strip()

    return datenstand

def read_measurement_log(filename):
    file = open(filename, 'r', encoding="utf8", errors='ignore')
    print(filename)
    logdata = file.read().splitlines()
    file.close()
    logdata = list(filter(None, logdata))

    return logdata

def read_measurement_csv(filename):
    file = open(filename, 'r', encoding="utf8", errors='ignore')
    print(filename)
    logdata = file.read().splitlines()
    file.close()
    df = list(filter(None, logdata))

    datenstand = [s for s in df if "Datenstand" in s]

    if len(datenstand) == 0:
        datenstand = [s for s in df if "DT_0169_RDBIPathDataSet" in s]

    if len(datenstand) == 0:
        datenstand = "nicht verfügbar"
    else:

        datenstand = str(datenstand[0]).split(";")[1]
     
    ecu = datenstand.split("-")[0]
    services_match = [s for s in df if "Services" in s]
    services_index = df.index(services_match[0])

    df = df[services_index:]
    number_of_scripts = str(df[0]).split(";")
    number_of_scripts = len([i for i in number_of_scripts if len(i) > 0])

    df = [[str(i).split(";")[j] for i in df if len(str(i).split(";")) > j] for j in range(0, number_of_scripts)]
    # df = [list(filter(None, i)) for i in df]

    df1 = list(chunk(df, 2))
    df1 = [list(x) for x in df1]
    services_names = [x[0][0].split(":")[0] for x in df1]

    df = [x[1:] for x in df]
    df = list(chunk(df, 2))
    df = [list(x) for x in df]

    dicts = [dict(zip(df[i][0], df[i][1])) for i in range(len(df))]
    df_concat = pd.concat([pd.DataFrame({'Signalname':list(dicts[i].keys()), 'value_orig':list(dicts[i].values()), 'Pruefstufe': services_names[i]}) for i in range(len(dicts))], axis = 0)
    df_concat['filename'] = filename


    df_concat['ECU'] = ecu
    df_concat['value'] = df_concat['value_orig'].apply(lambda x: pick_left_numbers(x))
    df_concat['unit'] =df_concat['value_orig'].apply(lambda x: pick_right_characters(x)) 

    return df_concat



app = Flask(__name__, instance_relative_config=False)

# app_dash = init_dashboard(app)

#UPLOAD_FOLDER = 'C:/Users/kluerman/Downloads'
#
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask(__name__)
app.secret_key = 'xyz'
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'csv', 'log'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_list_of_files_meeting_criterium(a, criterium_values):
    list_of_files = [i for i, j in enumerate(criterium_values) if j == a]
    return list_of_files


def get_list_of_files_meeting_criterium(a, criterium_values):
    list_of_files = [i for i, j in enumerate(criterium_values) if j == a]
    return list_of_files


def get_file_sublist(criterium_index, filename_list):
    # get list of files that have the same value for the given criterium
    # if multiple values for given criterium are present, make list of files for each criterium value

    criterium_values = [get_datenstand_csv(filename)[criterium_index + 1] for filename in filename_list
                        if ".csv" in filename]
    criterium_values_unique = list(dict.fromkeys(criterium_values))

    file_sublist = [get_list_of_files_meeting_criterium(a, criterium_values) for a in criterium_values_unique]

    return file_sublist, criterium_values_unique


def make_combinations(criterium_values_unique):
    combi = list(product(*criterium_values_unique))
    combi = [list(i) for i in combi]
    return combi



filenames_0 = []


@app.route('/upload', methods=['POST'])
def upload():

    try:

        files = []

        files = request.files.getlist('file')

            # print("show session filenames")
            # print(session['filenames'])

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print(file_path)

                filenames_0.append(file_path)
        print("show contents")
        print(os.listdir(app.config['UPLOAD_FOLDER']))

        return "OK"
    
    except: 
        [os.remove(filename) for filename in os.listdir(app.config['UPLOAD_FOLDER'])]
        return render_template('index.html')


@app.route('/upload_complete')
def upload_complete():
    
    print("show session filenames upload complete")
    filename_list = os.listdir(app.config['UPLOAD_FOLDER'])
    # filename_list = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(app.config['UPLOAD_FOLDER']) for filename in filenames 
    #        if (filename.endswith('.csv') | filename.endswith('.log'))]
    # filename_list = filename_list[:1]
    filename_list = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filename_list]
    print(filename_list)


    filename_checked = ["not checked"] * (len(filename_list) + 1)
    filename_checked[0] = "checked"


    datenstand = [get_datenstand_csv(filename) if ".csv" in filename
                    else get_datenstand_log(filename) for filename in filename_list]
    datenstand_unique = list(dict.fromkeys(datenstand))

    filename_list_lengths = [
            len([filename_list[index] for index, value in enumerate(datenstand) if value == datenstand_unique_i]) for
            datenstand_unique_i in datenstand_unique]

    list_entries = [f"Datenstand: {datenstand_unique[i]}   Anzahl Dateien: {filename_list_lengths[i]}" for i in
                        range(0, len(datenstand_unique))]

    return render_template('export.html',
                           list_of_datenstand=list_entries)


@app.route("/downloadCSV", methods=['GET', 'POST'])
def downloadCSV():
    filename_list = os.listdir(app.config['UPLOAD_FOLDER'])
    filename_list = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filename_list]

    if request.method == 'POST':

        criteria_selected = request.form.getlist('list_of_datenstand_selected[]')
        print("criteria")
        print(criteria_selected)

        criteria_selected = list(filter(None, criteria_selected))
        print(criteria_selected)

        criteria_selected = [str(i).split(" ")[1] for i in criteria_selected]

        datenstand = [get_datenstand_csv(filename) if ".csv" in filename
                    else get_datenstand_log(filename) for filename in filename_list]

        issubset_indices = [i for i, j in enumerate(datenstand) for criteria_selected_i in criteria_selected if
                                j == criteria_selected_i]
        issubset_indices = list(set(list(range(len(datenstand)))) - set(issubset_indices))

        print(issubset_indices)
        filename_sublist = [filename_list[i] for i in issubset_indices]
        print(filename_sublist)
        [os.remove(filename) for filename in filename_sublist]

        filename_list = os.listdir(app.config['UPLOAD_FOLDER'])
        filename_list = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filename_list]

        datenstand = [get_datenstand_csv(filename) if ".csv" in filename
                    else get_datenstand_log(filename) for filename in filename_list]

        print(filename_list)
        appended_data = export_csv(filename_list)

        [os.remove(filename) for filename in filename_list]
        print('files removed')
        print(os.listdir(app.config['UPLOAD_FOLDER']))
        outputname = datetime.datetime.now().strftime('Dauerlaufvergleich-%Y-%m-%d-%H-%M.csv')            
        return Response(
                appended_data.to_csv(sep=';'),
                mimetype="text/csv",
                headers={"Content-disposition":
                            "attachment; filename=" + outputname})

@app.route('/')
def input():
    print(os.listdir(app.config['UPLOAD_FOLDER']))
    filename_list = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in os.listdir(app.config['UPLOAD_FOLDER'])]
    [os.remove(filename) for filename in filename_list]

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 8080)
