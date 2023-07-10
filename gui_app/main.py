import string
import random
import os
import glob

from PIL import Image
from flask import Flask, request, render_template, session, url_for


app = Flask(__name__)
app.secret_key = b"testkey" # to do: get from env var

data_dir = "data/"  # to do: get from env var

def make_config():
    # to do: load some default values from file
    return {
        "image_size" : 1000,
        "descriptor_type": "sift"
    }

    

def make_session_id(length=10):
    return "".join(random.choices(string.ascii_lowercase,k=length))

@app.route("/")
def index():
    if not "session_id" in session:
        # create user session and data dir
        session["session_id"] = make_session_id()
        session["config"] = make_config()
        os.makedirs(f"""{data_dir}/{session["session_id"]}/""")
    
    return render_template("index.html", session=session)

@app.route("/reset")
def reset():
    # reset user session and data dir
    session = {}
    session["session_id"] = make_session_id()
    session["config"] = make_config()
    os.makedirs(f"""{data_dir}/{session["session_id"]}/""")
    
    return render_template("index.html", session=session)

from werkzeug.utils import secure_filename

def create_preview_img(file_path, size=(512,512)):
    im = Image.open(file_path)
    im.thumbnail(size, Image.ANTIALIAS)
    outfile = f"""{session["session_id"]}_preview.jpg"""
    im.save(f"{os.path.dirname(os.path.abspath(__file__))}/static/"+outfile, "JPEG")
    return url_for('static', filename=outfile)

@app.route('/upload', methods=['POST'])
def upload_file():
    print("upload")
    file = request.files['filename']
    print(file)
    file_path = f"""{data_dir}/{session["session_id"]}/{secure_filename(file.filename)}"""
    file.save(file_path)

    session["filename"] = file.filename
    session["preview_img"] = create_preview_img(file_path)

    return render_template("index.html", session=session)

@app.route('/set_index', methods=['POST'])
def set_index():
    print(list(request.form.items()))
    session["selected_index"] = request.form["selected_index"]
    return render_template("index.html", session=session)

@app.route('/run_georef', methods=['POST'])
def run_georef():
    print(list(request.form.items()))
    # update config params
    for key, value in request.form.items():
        session["config"][key] = value
    print("config to be used", session["config"])

    # call georef backend
    restrict_hypos = 3 # to do: should be in config
    sheets = "sampledata/blattschnitt_kdr100_fixed_dhdn.geojson" # to do: should be in index config
    input_image = f"""{data_dir}/{session["session_id"]}/{secure_filename(session["filename"])}""" # this removes umlaut
    cmd = f""" python main.py {input_image} {sheets} -r {restrict_hypos} """ # to do: import and run or call backend api instead
    os.system(cmd)

    outfile = "output/"#f"""{data_dir}/{session["id"]}/output""" # to do: set better output path for georeferencing script
    os.makedirs(outfile, exist_ok=True)
    # find result file
    base_filename = secure_filename(os.path.splitext(session["filename"])[0])
    print(base_filename)
    result_files = glob.glob(f"{outfile}/{base_filename}_aligned_*")
    print(result_files)

    # serve result
    result_image = next(filter(lambda x: x.split(".")[-1]=="jpg", result_files))
    result_preview_img = create_preview_img(result_image)
    return render_template("result.html", session=session, result_preview_img=result_preview_img)