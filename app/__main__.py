import cv2
import keras
import base64
from pathlib import Path
from flask import Flask
from flask import request
from flask import redirect
from flask import url_for
from flask import render_template
from flask import flash
from werkzeug.utils import secure_filename
from fruit_classifier.predict.__main__ import main


app = Flask(__name__)

# Make upload directory and allowed extensions visible in file scope
ROOT_DIR = Path(__file__).absolute().parents[1]
UPLOAD_DIR = ROOT_DIR.joinpath('upload_dir')
MODEL_FILES_DIR = ROOT_DIR.joinpath('model_files')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not UPLOAD_DIR.is_dir():
    UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
    app.logger.info(f'{UPLOAD_DIR} created')


app.config['UPLOAD_DIR'] = str(UPLOAD_DIR)

# http://flask.pocoo.org/docs/latest/quickstart/#sessions
# Secret needed for flash()
# WARNING: In real applications this needs to be kept secret
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def allowed_file(filename):
    """
    Checks whether the filename has an allowed extension

    Parameters
    ----------
    filename : str
        The filename as a string

    Returns
    -------
    bool
        Whether or not the filename is

    References
    ----------
    http://flask.pocoo.org/docs/latest/patterns/fileuploads/
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    The main page of the image classifier app

    Redirects to confirm_file when an allowed file is selected.
    Will redirect to itself if file is not selected properly.

    Returns
    -------
    Response
        For GET request index.html is rendered
        For POST where a valid file name is chosen, the user is
        redirected to confirm_file(), else a redirection to this
        function

    References
    ----------
    http://flask.pocoo.org/docs/latest/patterns/fileuploads/
    """
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # If user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # NOTE: Storing the file on server because:
            #       - The files can be too large to store in session
            #       - Base64 strings of the images can have more than
            #         the allowed 2083 characters for the URL
            #       - It's usually a bad idea to keep in memory for
            #         several parallel sessions
            file.save(str(Path(app.config['UPLOAD_DIR']).
                          joinpath(filename)))
            return redirect(url_for('confirm_file',
                                    filename=filename))

        else:
            flash('Not a valid file extension')
            return redirect(request.url)

    elif request.method == 'GET':
        # Clean uploaded files
        for f in UPLOAD_DIR.glob('**/*'):
            f.unlink()
        return render_template('index.html')


@app.route('/confirm/<filename>', methods=['GET'])
def confirm_file(filename):
    """
    Displays uploaded image and ask for confirmation to classify

    Parameters
    ----------
    filename : str
        The file name of the image

    Returns
    -------
    Response
        Displays confirm.html which gives the user the choice to
        start from scratch or classify the image
    """
    path = Path(app.config['UPLOAD_DIR']).joinpath(filename)

    # The image is encoded to base64 in order to display it
    image_b64 = base64.b64encode(path.read_bytes()).decode('utf-8')
    return render_template('confirm.html',
                           image_b64=image_b64,
                           filename=filename)


@app.route('/classify/<filename>', methods=['GET'])
def classify_file(filename):
    """
    Classifies the image and displays the result

    Parameters
    ----------
    filename : str
        The file name as a string

    Returns
    -------
    Response
        Displays the predicted image in prediction.html
        This web page also contains a button to start over
    """
    path = Path(app.config['UPLOAD_DIR']).joinpath(filename)

    # Clear any existing keras sessions and predict
    keras.backend.clear_session()
    output = main(path, MODEL_FILES_DIR)

    # Store the output image after converting to GBR
    cv2.imwrite(str(path), output[..., ::-1])

    # The image is encoded to base64 in order to display it
    result_b64 = base64.b64encode(path.read_bytes()).decode('utf-8')
    return render_template('prediction.html', result_b64=result_b64)


if __name__ == '__main__':
    # NOTE: Threaded is set to False to have the loaded model on the
    #       same thread
    #       See
    #       https://stackoverflow.com/questions/49400440/using-keras-model-in-flask-app-with-threading
    #       for details
    app.run(debug=False, host='0.0.0.0', port='5000', threaded=False)
