import string
import random
import os
import glob
import json
import subprocess
import re
import time
import shutil


from PIL import Image
from flask import Flask, request, render_template, session, url_for, send_from_directory, send_file

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = b"testkey" # to do: get from env var

data_dir = os.environ["DATA_DIR"]  # where to put all files that get created on the way

def make_config():
    """load default config values from file"""
    config_path = app.root_path+"/config.json"
    with open(config_path) as fr:
        config_data = {k:v.get("default") for k,v in json.load(fr).items()}
    return config_data


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


def create_preview_img(file_path, size=(512,512)):
    im = Image.open(file_path)
    im.thumbnail(size, Image.ANTIALIAS)
    outfile = f"""{session["session_id"]}_preview.jpg"""
    im.save(f"{app.root_path}/static/"+outfile, "JPEG")
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
    cmd = f""" python main.py {input_image} {sheets} -r {restrict_hypos} -v""" # to do: import and run or call backend api instead
    cmd = ["python",
            "main.py", 
            input_image,
            sheets,
            "-r", str(restrict_hypos),
            "-v"]
    # os.system(cmd)
    t0 = time.time()
    try:
        georef_output_log = subprocess.check_output(cmd).decode()
    except subprocess.CalledProcessError as err:
        georef_output_log = err
    t1 = time.time()

    print(georef_output_log)
    prediction = re.findall("(?<=pred: )[^ ]+", georef_output_log)[0]
    num_matches = re.findall("(?<=with score )[0-9]+", georef_output_log)[0]
    ecc_score = re.findall("(?<=with score: )[0-9.]+", georef_output_log)[0]

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
    res_dir = data_dir+"/"+session["session_id"]+"/results"
    if os.path.isdir(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    for file in result_files:
        shutil.move(file,res_dir)

    shutil.make_archive(app.root_path+"/static/"+session["session_id"]+"/result", 'zip', res_dir)

    results = {
        "preview_img": result_preview_img,
        "predicted_sheet_name": prediction,
        "num_matches": num_matches,
        "ecc_score": ecc_score,
        "time_taken": t1-t0,
        "bbox":result_image.split("_")[-1][:-4].split("-")
    }

    return render_template("result.html", session=session, results=results)

@app.route('/download', methods=['GET', 'POST'])
def download():
    # uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    return send_file("static/"+session["session_id"]+"/result.zip", as_attachment=True)