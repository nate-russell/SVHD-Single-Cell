'''
Author Nate Russell
ntrusse2@illinois.edu
'''
from __future__ import division
import argparse
import scipy
accelerator = True  # If True, turns on WebGL
from pprint import pprint
from bokeh.client import session, push_session
from bokeh.io import gridplot, curdoc, push, set_curdoc
from bokeh.models import BoxSelectTool, LassoSelectTool, HoverTool, WheelZoomTool, \
    PanTool, SaveTool, RedoTool, UndoTool, PolySelectTool, TapTool
from bokeh.models.layouts import WidgetBox
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import Button, DataTable, TableColumn, Markup, Paragraph
from matplotlib.cm import get_cmap
import numpy as np
import matplotlib as mpl
from bokeh.layouts import row,column,widgetbox
from bokeh.models import CustomJS,ColumnDataSource
from bokeh.models.widgets import Panel, Tabs
from time import time
import os
from collections import OrderedDict
import tempfile
import warnings
from functools import partial

mpl.rcParams['legend.markerscale'] = 2
from matplotlib import gridspec
from matplotlib.colors import rgb2hex
import random
import pandas as pd
import matplotlib.cm as cm
from bokeh.models.widgets import Select, TextInput, MultiSelect

from sklearn.datasets import make_blobs

from collections import defaultdict

def_quant_cmap = cm.get_cmap('viridis')
def_qual_cmap = cm.get_cmap('Paired')


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

def qual_2_color(X, dim_1=None, dim_2=None, cmap=def_qual_cmap):
    """

    :param X:
    :param dim_1:
    :param dim_2:
    :param cmap:
    :return:
    """
    if isinstance(X, np.ndarray):
        X = X.tolist()
    r = lambda: random.randint(0, 255)
    label = np.unique(X)
    m = len(label)

    norm = mpl.colors.Normalize(vmin=0, vmax=m-1)

    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(np.arange(m))
    rgba = rgba[:, 0:3]
    c = np.floor(rgba * 255)

    colors = ['#%02X%02X%02X' % (c[i, 0], c[i, 1], c[i, 2]) for i in np.arange(m)]

    mapper = dict(zip(label, colors))

    pprint(mapper)

    color_list = np.array([mapper.get(i) for i in X])

    return color_list

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb[0:3]

def quant_2_color(x, n_colors=5, cmap=def_quant_cmap, robust='none', limit=3):
    """

    :param x:
    :param n_colors:
    :param cmap:
    :param robust:
    :param limit:
    :return:
    """
    if isinstance(x, np.ndarray):
        x = x.tolist()
    assert isinstance(x,list)

    unique = np.unique(x)

    if len(unique) == 2:
        return ['#CC3300' if xi == unique[0] else '#0066CC' for xi in x]
    else:
        if robust == 'percentile':
            norm = mpl.colors.Normalize(vmin=np.percentile(x, 2), vmax=np.percentile(x, 98))
        elif robust == 'std':
            stdv = np.std(x)
            mu = np.mean(x)
            norm = mpl.colors.Normalize(vmin=max([min(x), (mu - (limit * stdv))]),
                                        vmax=min([max(x), (mu + (limit * stdv))]))
        else:
            norm = mpl.colors.Normalize(vmin=min(x), vmax=max(x))

        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba = mapper.to_rgba(x)
        rgba = rgba[:, 0:3]
        c = np.floor(rgba * 255)

        color_list = ['#%02X%02X%02X' % (c[i, 0], c[i, 1], c[i, 2]) for i in range(len(x))]

        return color_list

def get_color_map(c):
    """

    :param x:
    :return:
    """
    if isinstance(c[0], str) == False:
        color_list = quant_2_color(c, cmap=def_quant_cmap)
    else:
        color_list = qual_2_color(c, cmap=def_qual_cmap)

    return color_list

class GPE:

    def __init__(self,offline=False,test_n_plots=6,test_n_samples=1000,max_row_width=4):
        """
        Initialize Graph Projection Explorer
        """
        print("\n\n" + '#' * 75)
        print("Bokeh Graph Projection Explorer V.0.0.1")
        print('Class Format')
        print('#' * 75 + '\n\n')

        self.test_n_plots = test_n_plots
        self.test_n_samples = test_n_samples
        self.max_row_width = max_row_width
        self.testmode = False
        self.n_newplots = 0


        if offline:
            # Operate in non command line argument mode
            self.testmode = True
            self.verbose = True
        else:
            parser = argparse.ArgumentParser()
            parser.add_argument("--dir", help="directory housing data")
            parser.add_argument("--mode", help="Options: Test, Presentation, Default")
            parser.add_argument("--downsample", help="If provided, randomly samples the data the provided number of times")
            parser.add_argument("--verbose", help="If True, Prints messages to console where server is running")
            self.args = parser.parse_args()


            try:
                self.args.downsample = int(self.args.downsample)
            except:
                TypeError()



            if self.args.verbose:
                self.verbose = True
                print "Verbose Turned On"

            if isinstance(self.args.mode, str):
                if self.args.mode.lower() == 'test':
                    if self.verbose:
                        print('Test Mode Selected')
                        self.testmode = True
                    else:
                        raise ValueError("Mode argument: " + self.args.mode + "Not among valid modes")
            else:
                raise ValueError("Mode Argument must be of type str. Instead recieved: " + str(type(self.args.mode)))

        # Build Test Data if it exists
        if self.testmode:
            self.data_dir = self.gen_test_data()
        else:
            self.data_dir = self.args.dir


        # Initialize
        self.init_data()
        self.init_color()
        self.init_controls()
        if self.verbose:
            print("Initialization Timings")
            print("\tData Init:    "+str(self.init_data_time))
            print("\tColor Init:   " + str(self.init_color_time))
            print("\tControl Init: " + str(self.init_control_time))

    def read_data(self):
        """

        :return:
        """
        if self.verbose: print('\nReading Data\n')

        # Check Data
        assert os.path.isdir(self.data_dir)
        vecdir = os.path.join(self.data_dir,'vectors')
        graphdir = os.path.join(self.data_dir,'graphs')
        assert  os.path.isdir(vecdir)
        try:
            assert os.path.isdir(graphdir)
        except AssertionError:
            pass

        vec_files = [os.path.join(vecdir,file) for file in os.listdir(vecdir) if file.endswith(".csv")]
        vec_files.sort()

        assert len(vec_files) > 0

        self.plot_df = OrderedDict()
        self.data_df = OrderedDict()
        self.maps_dict = OrderedDict()
        self.true_cols = []
        self.initial_plot_2_data_mapper = {}

        n_plot = 1
        self.n_plots = 0
        for i,f in enumerate(vec_files):

            if self.verbose: print("Reading File: %s"%f)

            file_prefix = f.split('/')[-1].split('.')[0] + "_"

            df = pd.read_csv(f)
            # Sometimes Unnamed: 0 creeps in by mistake of the user
            if "Unnamed: 0" in df.columns:
                df.drop(["Unnamed: 0"],axis=1,inplace=True)

            # Confirm Data Dim
            if i == 0:
                self.n,p = df.shape
            else:
                n,p = df.shape
                #assert n == self.n

            if isinstance(self.args.downsample, int):
                if self.verbose: print("Downsampling: %d"%self.args.downsample)
                df = df.sample(n=int(self.args.downsample),replace=False,random_state=1,axis=0)
                print(df.shape)
                self.n, p = df.shape

            # Test if D1 and D2 columns are found
            has_d1 = 0
            has_d2 = 0
            for col in df.columns:
                if 'D1' in col:
                    has_d1 += 1
                elif 'D2' in col:
                    has_d2 += 1
            if has_d1 == 1 and has_d2 == 1:
                has_both = True
            else:
                has_both = False

            if has_d1 > 1:
                warnings.warn("Too many column headers contain D1, cannot disambiguate")
            if has_d2 > 1:
                warnings.warn("Too many column headers contain D2, cannot disambiguate")
            if has_d1 != has_d2:
                warnings.warn("The number of D1 and D2 variable do not match")

            # Now that data validation is done, actually add data to self.df
            for col in df.columns:

                if 'D1' in col and has_both:
                    self.data_df[file_prefix + col] = df[col].values.tolist()
                    self.plot_df['Plot_%d_x' % n_plot] = df[col].values.tolist()
                    self.plot_df[file_prefix + col] = df[col].values.tolist()
                    self.true_cols.append(file_prefix + col)
                    self.initial_plot_2_data_mapper['Plot_%d_x' % n_plot] = file_prefix + col


                elif 'D2' in col and has_both:
                    self.data_df[file_prefix + col] = df[col].values.tolist()
                    self.plot_df['Plot_%d_y' % n_plot] = df[col].values.tolist()
                    self.plot_df[file_prefix + col] = df[col].values.tolist()
                    self.true_cols.append(file_prefix + col)
                    self.initial_plot_2_data_mapper['Plot_%d_y' % n_plot] =  file_prefix + col

                else:
                    self.data_df[file_prefix + col] = df[col].values.tolist()
                    self.plot_df[file_prefix + col] = df[col].values.tolist()
                    self.true_cols.append(file_prefix + col)

            if has_both:
                self.maps_dict["Plot_%d" % n_plot] = ('Plot_%d_x' % n_plot,'Plot_%d_y' % n_plot)
                n_plot += 1

    def init_data(self):
        """
        Load and Validate Data
        :return:
        """

        t0 = time()

        if self.verbose: print("Initializing Data Resources")
        self.read_data()


        self.n_plots = len(self.maps_dict.keys())
        self.color = "__COLOR__"
        self.plot_df["__COLOR__"] = ["#80bfff"] * self.n
        self.plot_df["__selected__"] = np.ones_like(self.n, dtype=np.bool).tolist()

        self.data_df = pd.DataFrame(self.data_df)
        self.data_dict = self.data_df.to_dict(orient='list')
        assert isinstance(self.data_dict, dict)

        self.plot_df = pd.DataFrame(self.plot_df)
        self.plot_dict = self.plot_df.to_dict(orient='list')
        assert isinstance(self.data_dict, dict)

        # Used for indexing Selected Data
        self.inds_bool = np.ones_like(np.arange(self.n), dtype=np.bool)
        self.source = ColumnDataSource(data=self.plot_dict)
        self.table_source = ColumnDataSource(data=self.plot_df[self.true_cols].to_dict(orient='list'))

        self.init_data_time = time() - t0
        return self.init_data_time

    def gen_test_data(self):
        """
        Generate Test Data, Store in temp dir and return dir path
        :return: dir path
        """
        if self.verbose: print('Generating Test Data')

        # Initialize Temp Testing Dir Structure
        tmpdir = tempfile.mkdtemp()
        tmpdir_p = os.path.join(tmpdir,'vectors')
        tmpdir_g = os.path.join(tmpdir,'graphs')
        os.mkdir(tmpdir_p)
        os.mkdir(tmpdir_g)

        assert isinstance(self.test_n_plots, int)
        assert isinstance(self.test_n_samples, int)

        # Make Blob data
        X, y = make_blobs(n_samples=self.test_n_samples, n_features=self.test_n_plots * 2,
                          centers=6, cluster_std=0.75, random_state=1)


        # Store blob data in test dir
        for i in range(self.test_n_plots):


            cols = X[:, (i * 2):((i * 2) + 2)]
            #cols[np.random.choice(range(self.test_n_samples),1),:] = [np.null,np.null]

            df = pd.DataFrame(data=cols, columns=('D1','D2'))
            df.to_csv(os.path.join(tmpdir_p,'P%d.csv'%i))
        meta_df = pd.DataFrame({'Meta':['Class: ' + str(label) for label in y]})
        meta_df.to_csv(os.path.join(tmpdir_p, 'Meta.csv'))

        # Generate Graph Data
        # TODO

        return tmpdir

    def init_color(self):
        """

        :return:
        """
        t0 = time()
        if self.verbose: print("Initializing Color Resources")
        self.color_map_dict = {col: list(enumerate(get_color_map(self.data_dict[col]))) for col in self.data_dict.keys()}
        self.init_color_time = time() - t0
        return self.init_color_time

    def init_controls(self):
        """

        :return:
        """
        t0 = time()
        if self.verbose: print("Initializing Controls")


        # Initialize Controls
        self.color_selection = Select(title="Color By", options=self.data_dict.keys(), value=self.data_dict.keys()[0])
        self.selection_label = TextInput(value="MyGroup#1", title="Selection Label:")

        self.add_selection_label = Button(label="Add Selection Label")
        self.write_mod_file = Button(label="Download", button_type="primary")
        self.write_mod_file.callback = CustomJS(args=dict(source=self.source),
                                                code=open(os.path.join(os.path.dirname(__file__),
                                                                       "download.js")).read())

        self.tooltip_select = MultiSelect(title='Tooltip',value = [self.data_dict.keys()[0]],
                                          options=[(key,key.upper()) for key in self.data_dict.keys()])

        # Declare Tooltip Contents
        self.tooltip_list = [(col, "@" + col) for col in self.tooltip_select.value]

        self.init_control_time = time() - t0
        return self.init_control_time

    def add_selection(self):
        """
        Add new column to source containing copy of selection
        :return:
        """
        self.source.add(self.source.data['__selected__'],name=self.selection_label.value)





    def make_plot(self,title, x, y):
        """

        :param title:
        :param x:
        :param y:
        :return:
        """

        print(title,x,y)


        t0 = time()

        pt = PanTool()
        lst = LassoSelectTool()
        pst = PolySelectTool()
        bst = BoxSelectTool()
        wzt = WheelZoomTool()
        tt = TapTool()
        st = SaveTool()
        ut = UndoTool()
        rt = RedoTool()

        p = figure(
            tools=[pt,lst,pst,bst,wzt,tt,st,ut,rt],
            plot_width=400,
            plot_height=400,
            title=self.initial_plot_2_data_mapper[x]+" vs. "+self.initial_plot_2_data_mapper[y],
            webgl=accelerator)
        # configure so that no drag tools are active
        p.toolbar.active_drag = pt

        # configure so that Bokeh chooses what (if any) scroll tool is active
        p.toolbar.active_scroll = wzt

        # configure so that a specific PolySelect tap tool is active
        p.toolbar.active_tap = tt

        p.xaxis.axis_label = self.initial_plot_2_data_mapper[x]
        p.yaxis.axis_label = self.initial_plot_2_data_mapper[y]
        c = p.circle(x=x, y=y, size=5, color="__COLOR__", alpha=.75, source=self.source,
                     hover_color='white', hover_alpha=1, hover_line_color='grey')
        c.data_source.on_change('selected', self.update)


        # Edge generator
        '''
        self.graph_set = [{i: [[1,0.15],[2,0.5],[3,0.99]] for i in range(self.n)}]

        self.edge_colors = qual_2_color(['g'+str(i) for i,_ in enumerate(self.graph_set)])

        self.edge_sources = [ColumnDataSource({'x0': [],
                                               'y0': [],
                                               'x1': [],
                                               'y1': [],
                                               'alpha': []})
                             for i in self.graph_set]

        self.edge_segments = [p.segment(x0='x0',
                                        y0='y0',
                                        x1='x1',
                                        y1='y1',
                                        color=self.edge_colors[i],
                                        alpha='alpha',
                                        line_width=3,
                                        #line_dash=[1,1],
                                        source=self.edge_sources[i])
                              for i, _ in enumerate(self.graph_set)]

        for i, _ in enumerate(self.graph_set):
            code1 = """
                    var links = %s;
                    var data = {'x0': [], 'y0': [], 'x1': [], 'y1': [], 'alpha': []};
                    var cdata = circle.get('data');
                    var indices = cb_data.index['1d'].indices;
                    for (i=0; i < indices.length; i++) {
                    ind0 = indices[i]
                    for (j=0; j < links[ind0].length; j++) {
                    ind1 = links[ind0][j][0];
                    w = links[ind0][j][1];
                    """ % self.graph_set[i]
            code2 = "data['x0'].push(cdata['" + x + "'][ind0]);\n" + \
                    "data['y0'].push(cdata['" + y + "'][ind0]);\n" + \
                    "data['x1'].push(cdata['" + x + "'][ind1]);\n" + \
                    "data['y1'].push(cdata['" + y + "'][ind1]);\n" + \
                    "data['alpha'].push([w]);\n"
            code3 = "}}segment.set('data', data);"
            code = code1 + code2 + code3
            callback = CustomJS(args={'circle': c.data_source,
                                      'segment': self.edge_segments[i].data_source},
                                code=code)
            p.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[c]))
        '''

        p.select(BoxSelectTool).select_every_mousemove = False
        p.select(LassoSelectTool).select_every_mousemove = False



        # Plot Controls
        xdim_select = Select(title="X Dim", options=self.data_dict.keys(), value=self.initial_plot_2_data_mapper[x],width=400)
        ydim_select = Select(title="Y Dim", options=self.data_dict.keys(), value=self.initial_plot_2_data_mapper[y],width=400)
        xdim_select.on_change('value', self.plot_update)
        ydim_select.on_change('value', self.plot_update)
        remove = Button(label="Remove", button_type="danger",width=400)
        remove.on_click(partial(self.remove_plot,title,x,y))

        self.plot_control_dict[title] = {'x':xdim_select,
                                         'y':ydim_select,
                                         'xprev':xdim_select.value,
                                         'yprev':ydim_select.value,
                                         'figure':p,
                                         'tooltip':HoverTool(tooltips=self.tooltip_list,point_policy='snap_to_data',show_arrow=False)}
        # Give the hover tool a tool tip
        self.plot_control_dict[title]['figure'].add_tools(self.plot_control_dict[title]['tooltip'])


        # Form Tab
        plot_options = WidgetBox(xdim_select,ydim_select,remove)
        tab1 = Panel(child=self.plot_control_dict[title]['figure'], title=title,width=400,height=400)
        tab2 = Panel(child=plot_options, title="options",width=400,height=400)
        tabs = Tabs(tabs=[tab1, tab2],width=400,height=400)


        self.tab_list.append(tabs)
        self.circle_list.append(c)

        print('Plot Time: ' + str(time() - t0))

        return tabs, c

    def change_color(self,attrname, old, new):
        """
        Change Color
        :return:
        """
        if self.verbose: print("---self.change_color---")
        self.source.patch({"__COLOR__": self.color_map_dict[self.color_selection.value]})
        if self.verbose: print('New Color: '+self.color_selection.value)

    def refresh_plots(self):
        """
        Refresh plots
        :return:
        """
        if self.verbose: print('---self.refresh_plots---')
        print(self.layout)
        print(self.layout.children)
        self.layout.children[0] = self.make_all_plots()
        print(self.layout)
        print(self.layout.children)

        #controls = self.make_all_controls()
        #dt = self.make_data_table()
        #self.layout.children[1] = row(controls, dt)

    def add_plot(self):
        """
        Add new plot to the
        :return:
        """
        if self.verbose: print('---self.add_plot---')
        self.n_newplots += 1


        d1 = self.true_cols[np.random.randint(len(self.true_cols))]
        d2 = self.true_cols[np.random.randint(len(self.true_cols))]

        self.source.add(self.source.data[d1],name='NewPlot%d_x'%self.n_newplots)
        self.source.add(self.source.data[d2], name='NewPlot%d_y'%self.n_newplots)
        self.maps_dict['NewPlot%d'%self.n_newplots] = ('NewPlot%d_x'%self.n_newplots,
                                                       'NewPlot%d_y'%self.n_newplots)

        self.initial_plot_2_data_mapper['NewPlot%d_x'%self.n_newplots] = d1
        self.initial_plot_2_data_mapper['NewPlot%d_y'%self.n_newplots] = d2

        if self.verbose: print('---self.add_plot---Done')
        self.refresh_plots()

    def remove_plot(self,plot_title,x,y):
        """
        Removes a plot
        :return:
        """
        if self.verbose: print('\n---self.remove_plot---')

        print('Getting Rid of')
        print(self.maps_dict[plot_title])
        self.maps_dict.pop(plot_title)
        print(x,y)
        #self.source.data.pop(x)
        #self.source.data.pop(y)

        print('After poping')
        print(self.maps_dict)
        print(self.source.data.keys())
        pprint(self.source.data)



        self.refresh_plots()


    def dist_plot(self):
        """

        :return:
        """
        dist_plots = []

        self.dist_dict = {}

        for col in self.true_cols[0:8]:

            x = self.data_dict[col]

            # create the horizontal histogram
            hhist, hedges = np.histogram(x, bins=20)
            hzeros = np.zeros(len(hedges) - 1)
            hmax = max(hhist) * 1.1

            LINE_ARGS = dict(color="#3A5785", line_color=None)

            ph = figure(toolbar_location=None, plot_width=200, plot_height=100,title=col,
                        y_range=(0, hmax), y_axis_location="left")
            ph.xgrid.grid_line_color = None
            ph.yaxis.major_label_orientation = np.pi / 4
            ph.background_fill_color = "#fafafa"

            ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
            hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)


            # Add to dist dict
            self.dist_dict[col] = {'plot':ph,
                                   'hh1':hh1,
                                   'hedges':hedges}


            # Add to grid plot
            dist_plots.append(ph)

        gp = gridplot(dist_plots, plot_width=300, plot_height=200, ncols=4, height=400,toolbar_location=None,)


        return gp




    def make_all_plots(self):
        """

        :return:
        """
        self.tab_list = []
        self.plot_list = []
        self.circle_list = []
        self.plot_control_dict = {}

        self.n_plots = len(self.maps_dict.keys())

        # Make Each Plot
        for f in self.maps_dict.keys():
            xs = self.maps_dict[f][0]
            ys = self.maps_dict[f][1]
            self.make_plot(f, xs, ys)

        #if self.verbose: print("Grid of plots: "+str(self.tab_list))
        return gridplot(self.tab_list,ncols=self.max_row_width,plot_width=250, plot_height=250)

    def make_all_controls(self):
        # Controls

        self.addplot = Button(label="Add New Plot", button_type="success")
        self.addplot.on_click(self.add_plot)

        self.add_selection_label.on_click(self.add_selection)
        self.tooltip_select.on_change('value',self.tooltip_update)
        self.color_selection.on_change('value',self.change_color)


        controls = [self.color_selection,
                    self.tooltip_select,
                    self.selection_label,
                    self.add_selection_label,
                    self.addplot,
                    self.write_mod_file]

        return widgetbox(children=controls)

    def pseudo_update(self,n_selected):
        """

        :param n_selected:
        :return:
        """
        assert self.n >= n_selected
        pseudo_selected = np.random.choice(np.arange(self.n),size=n_selected)
        pseudo_new = {'1d':{'indices':pseudo_selected}}
        self.update(1,1,pseudo_new)

    def tooltip_update(self,attrname, old, new):
        """
        Updates tooltip content for each plot
        :param attrname:
        :param old:
        :param new:
        :return:
        """

        if self.verbose: print("\n------self.tooltip_update-----")
        self.tooltip_list = [(col.encode("utf-8"), "@" + col.encode("utf-8")) for col in self.tooltip_select.value]
        self.refresh_plots()
        #if self.verbose: print(contents)

        #for p in self.plot_control_dict:
        #    self.plot_control_dict[p]['tooltip'].tooltips = contents
        #    self.plot_control_dict[p]['tooltip'].plot = self.plot_control_dict[p]['figure']
        #    if self.verbose: print(self.plot_control_dict[p]['tooltip'].tooltips)

    def plot_update(self,attrname, old, new):
        """
        Updates plot contents, title and axis labels
        :param attrname:
        :param old:
        :param new:
        :return:
        """
        # modify column value of data source

        if self.verbose: print("\n------self.plot_update-----")

        for p in self.plot_control_dict:

            xy_mod = False
            # X
            if self.plot_control_dict[p]['x'].value != self.plot_control_dict[p]['xprev']:
                if self.verbose: print('X change on %s'%p)
                # patch
                self.source.patch({p+"_x": list(enumerate(self.data_dict[self.plot_control_dict[p]['x'].value]))})
                # update prev
                self.plot_control_dict[p]['xprev'] = self.plot_control_dict[p]['x'].value
                self.plot_control_dict[p]['figure'].xaxis.axis_label = self.plot_control_dict[p]['x'].value
                xy_mod = True


            # Y
            if self.plot_control_dict[p]['y'].value != self.plot_control_dict[p]['yprev']:
                if self.verbose: print('Y change on %s' % p)
                # patch
                self.source.patch({p+"_y": list(enumerate(self.data_dict[self.plot_control_dict[p]['y'].value]))})
                # update prev
                self.plot_control_dict[p]['yprev'] = self.plot_control_dict[p]['y'].value
                self.plot_control_dict[p]['figure'].yaxis.axis_label = self.plot_control_dict[p]['y'].value
                xy_mod = True


            # update title text
            if xy_mod:
                self.plot_control_dict[p]['figure'].title.text = self.plot_control_dict[p]['x'].value + " vs. " + self.plot_control_dict[p]['y'].value

    def update(self,attrname, old, new):
        """

        :param attrname:
        :param old:
        :param new:
        :return:
        """
        if self.verbose: print("\n------self.update-----")

        t0 = time()

        # Update Selected
        try:
            inds = np.array(new['1d']['indices'])

            if len(inds) == 0 or len(inds) == self.n:
                print("NOTHING SELECTED")
                pass
            else:
                print('Selected Set Size: ' + str(len(inds)))

                # Modify Dist Plots
                if self.verbose: print('\tUpdating DistPlots')
                for dist in self.dist_dict:
                    x = np.array(self.data_dict[dist])
                    xbins = self.dist_dict[dist]['hedges']
                    hhist1, _ = np.histogram(x[inds], bins=xbins)
                    self.dist_dict[dist]['hh1'].data_source.data["top"] = hhist1

                # Modify Data Table
                indbool = self.inds_bool
                indbool[inds] = False
                self.source.patch({"__selected__":list(enumerate(indbool))})







                full_table_dict = self.data_dict
                self.table_source.data = {col: np.array(full_table_dict[col])[inds] for col in full_table_dict.keys()}

        # Hack to stop string index type error that occurs when you change color
        except FloatingPointError:
            print("NOTHING SELECTED, error caught")
            pass

        self.update_time = time()-t0
        print(self.update_time)

    def make_data_table(self):

        # Add Table
        columns = [TableColumn(field=col, title=col) for col in self.data_df.keys()]
        dt = DataTable(source=self.table_source,
                       columns=columns,
                       width=1800,
                       height=400,
                       scroll_to_selection=False)

        return WidgetBox(dt)

    def go(self):
        """

        :return:
        """

        print("\n\n" + '#' * 75)
        print("Server Engaged")
        print('#' * 75 + '\n\n')


        plots = self.make_all_plots()
        controls = self.make_all_controls()
        dt = self.make_data_table()
        dp = self.dist_plot()

        dp = gridplot(dp, plot_width=300, plot_height=200, ncols=2, height=400)
        dp = Panel(child=dp, title="Distribution Plot", width=400, height=400)

        dt = Panel(child=dt, title="Data Table", width=400, height=400)

        dtdp = Tabs(tabs=[dp,dt],width=400,height=400)


        self.layout =  column(children=[plots,row(controls,dtdp)])
        curdoc().add_root(self.layout)
        curdoc().title = 'Graph Projection Explorer'

