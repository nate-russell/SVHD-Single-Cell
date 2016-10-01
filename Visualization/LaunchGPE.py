import subprocess
import webbrowser
import os


def launch_localhost(dir,mode='normal',verbose=True):
    """
    Initializes a localhost Bokeh server
    :param dir: Parent Directory , Required subdirs: Vectors, Optional subdirs: Graphs
    :param mode: 'normal':runs the server with provided data   'test':runs a test version of the bokeh server with synthetic data
    :param verbose: if True, server will send status information to standard out
    :return: None
    """
    if not isinstance(mode,str): raise ValueError("param \"mode\" must be string")
    if not isinstance(dir, str): raise ValueError("param \"dir\" must be string and is a valid accesible path")
    if not isinstance(mode, bool): raise ValueError("param \"mode\" must be boolean")


    # Opens the server inside a browser if possible
    try:
        webbrowser.open("http://localhost:5006/GPE?webgl=1", new=1,autoraise=True)
    except:
        pass

    call_list = ["bokeh", "serve",os.path.join(os.path.dirname(__file__),"BokehTool"),
                     "--args",
                     "--mode",mode,
                     "--verbose",str(verbose)]
    print(call_list)
    subprocess.call(call_list)



